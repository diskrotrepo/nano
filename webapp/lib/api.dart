import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

enum NanoMode { generate, extend }

extension NanoModeX on NanoMode {
  String get path => switch (this) {
        NanoMode.generate => '/generate',
        NanoMode.extend => '/extend',
      };
  String get label => switch (this) {
        NanoMode.generate => 'generate',
        NanoMode.extend => 'extend',
      };
}

/// A picked audio file (bytes + filename), platform-agnostic.
class AudioFile {
  AudioFile(this.name, this.bytes);
  final String name;
  final Uint8List bytes;
}

/// Every tunable the inference server exposes, across all three endpoints.
/// Fields not relevant to the active mode are simply omitted when building
/// the request.
class GenParams {
  // conditioning (all modes)
  String prompt = '';
  String lyrics = '';
  String negativePrompt = '';
  bool sweeten = true;
  AudioFile? styleAudio;
  double styleWeight = 0.5;

  // sampling defaults — noise-resistant baseline (single source of truth so
  // the per-slider reset buttons can restore them).
  static const double defaultTemperature = 0.8;
  static const int defaultTopK = 120;
  static const double defaultTopP = 0.95;
  static const double defaultCfgScale = 4.0;
  static const double defaultLyricCfgScale = 3.0;

  // sampling (all modes)
  double temperature = defaultTemperature;
  int topK = defaultTopK;
  double topP = defaultTopP;
  String perCbTemperature = '';
  String perCbTopK = '';
  String perCbTopP = '';
  double cfgScale = defaultCfgScale;
  double lyricCfgScale = defaultLyricCfgScale;

  // generate-only
  double seconds = 30.0;
  bool scoreClap = false;

  // extend
  AudioFile? inputAudio;
  double addSeconds = 20.0;
  double overlapSeconds = 8.0;
}

class NanoResult {
  NanoResult({
    required this.bytes,
    required this.mime,
    this.sweetened,
    this.clapScore,
  });
  final Uint8List bytes;
  final String mime;
  final String? sweetened;
  final double? clapScore;
}

class HealthInfo {
  HealthInfo(this.raw);
  final Map<String, dynamic> raw;

  String get device => '${raw['device']}';
  String get ckptPath => '${raw['ckpt_path']}';
  int get ckptStep => (raw['ckpt_step'] ?? -1) as int;
  int get modelParams => (raw['model_params'] ?? 0) as int;
  bool get textConditioning => raw['text_conditioning'] == true;
  Map<String, dynamic> get codec =>
      (raw['codec'] as Map?)?.cast<String, dynamic>() ?? const {};
}

class NanoApi {
  NanoApi(this.baseUrl);
  String baseUrl;

  Uri _uri(String path) {
    var b = baseUrl.trim();
    if (b.endsWith('/')) b = b.substring(0, b.length - 1);
    return Uri.parse('$b$path');
  }

  Future<HealthInfo> health() async {
    final resp = await http.get(_uri('/health'));
    if (resp.statusCode != 200) {
      throw NanoApiException('health ${resp.statusCode}: ${resp.body}');
    }
    return HealthInfo(jsonDecode(resp.body) as Map<String, dynamic>);
  }

  Future<NanoResult> run(NanoMode mode, GenParams p) async {
    final req = http.MultipartRequest('POST', _uri(mode.path));
    final f = req.fields;

    // shared conditioning
    f['prompt'] = p.prompt;
    f['lyrics'] = p.lyrics;
    f['negative_prompt'] = p.negativePrompt;
    f['sweeten'] = p.sweeten.toString();
    f['style_weight'] = p.styleWeight.toString();

    // shared sampling
    f['temperature'] = p.temperature.toString();
    f['top_k'] = p.topK.toString();
    f['top_p'] = p.topP.toString();
    f['per_cb_temperature'] = p.perCbTemperature;
    f['per_cb_top_k'] = p.perCbTopK;
    f['per_cb_top_p'] = p.perCbTopP;
    f['cfg_scale'] = p.cfgScale.toString();
    f['lyric_cfg_scale'] = p.lyricCfgScale.toString();

    if (p.styleAudio != null) {
      req.files.add(http.MultipartFile.fromBytes(
        'style_audio', p.styleAudio!.bytes,
        filename: p.styleAudio!.name,
      ));
    }

    switch (mode) {
      case NanoMode.generate:
        f['seconds'] = p.seconds.toString();
        f['score_clap'] = p.scoreClap.toString();
      case NanoMode.extend:
        _requireInput(p);
        f['add_seconds'] = p.addSeconds.toString();
        f['overlap_seconds'] = p.overlapSeconds.toString();
        req.files.add(http.MultipartFile.fromBytes(
          'audio', p.inputAudio!.bytes,
          filename: p.inputAudio!.name,
        ));
    }

    final streamed = await req.send();
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode != 200) {
      throw NanoApiException(
        '${mode.label} ${resp.statusCode}: ${_errBody(resp.body)}',
      );
    }

    final sweet = resp.headers['x-nano-sweetened-prompt'];
    final clapRaw = resp.headers['x-nano-clap-score'];
    return NanoResult(
      bytes: resp.bodyBytes,
      mime: resp.headers['content-type'] ?? 'audio/mpeg',
      sweetened: (sweet != null && sweet.trim().isNotEmpty) ? sweet : null,
      clapScore: clapRaw != null ? double.tryParse(clapRaw) : null,
    );
  }

  void _requireInput(GenParams p) {
    if (p.inputAudio == null) {
      throw NanoApiException('this mode needs an input audio file');
    }
  }

  String _errBody(String body) {
    try {
      final j = jsonDecode(body);
      if (j is Map && j['detail'] != null) return '${j['detail']}';
    } catch (_) {}
    return body;
  }
}

class NanoApiException implements Exception {
  NanoApiException(this.message);
  final String message;
  @override
  String toString() => message;
}
