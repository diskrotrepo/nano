import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import 'api.dart';
import 'audio_web.dart';
import 'theme.dart';
import 'widgets.dart';

void main() => runApp(const NanoApp());

class NanoApp extends StatelessWidget {
  const NanoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'nano',
      debugShowCheckedModeBanner: false,
      theme: buildNanoTheme(),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _params = GenParams();
  NanoMode _mode = NanoMode.generate;

  final _serverCtl = TextEditingController(text: 'http://127.0.0.1:8000');
  final _promptCtl = TextEditingController();
  final _lyricsCtl = TextEditingController();
  final _negCtl = TextEditingController();
  final _perTempCtl = TextEditingController();
  final _perTopKCtl = TextEditingController();
  final _perTopPCtl = TextEditingController();

  bool _showAdvanced = false;
  bool _busy = false;
  String? _error;
  HealthInfo? _health;
  bool _healthLoading = false;

  NanoResult? _result;
  AudioBlob? _blob;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _checkHealth());
  }

  @override
  void dispose() {
    _blob?.revoke();
    for (final c in [
      _serverCtl, _promptCtl, _lyricsCtl, _negCtl,
      _perTempCtl, _perTopKCtl, _perTopPCtl,
    ]) {
      c.dispose();
    }
    super.dispose();
  }

  NanoApi get _api => NanoApi(_serverCtl.text);

  Future<void> _checkHealth() async {
    setState(() {
      _healthLoading = true;
      _health = null;
    });
    try {
      final h = await _api.health();
      if (mounted) setState(() => _health = h);
    } catch (_) {
      if (mounted) setState(() => _health = null);
    } finally {
      if (mounted) setState(() => _healthLoading = false);
    }
  }

  Future<AudioFile?> _pickAudio() async {
    final res = await FilePicker.pickFiles(
      type: FileType.audio,
      withData: true,
    );
    final f = res?.files.firstOrNull;
    if (f?.bytes == null) return null;
    return AudioFile(f!.name, f.bytes!);
  }

  Future<void> _run() async {
    // sync controllers into params
    _params
      ..prompt = _promptCtl.text
      ..lyrics = _lyricsCtl.text
      ..negativePrompt = _negCtl.text
      ..perCbTemperature = _perTempCtl.text
      ..perCbTopK = _perTopKCtl.text
      ..perCbTopP = _perTopPCtl.text;

    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final result = await _api.run(_mode, _params);
      _blob?.revoke();
      final blob = AudioBlob.fromBytes(result.bytes, result.mime);
      if (!mounted) return;
      setState(() {
        _result = result;
        _blob = blob;
      });
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 640),
          child: ListView(
            padding: const EdgeInsets.fromLTRB(20, 28, 20, 60),
            children: [
              _header(),
              const SizedBox(height: 22),
              _modeSelector(),
              const SizedBox(height: 18),
              _conditioningCard(),
              _samplingCard(),
              _modeCard(),
              _runButton(),
              if (_error != null) _errorBox(),
              if (_result != null && _blob != null) _resultCard(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _header() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          crossAxisAlignment: CrossAxisAlignment.baseline,
          textBaseline: TextBaseline.alphabetic,
          children: [
            const Text(
              'nano',
              style: TextStyle(
                fontSize: 34,
                fontWeight: FontWeight.w900,
                letterSpacing: -1,
                color: NanoColors.text,
              ),
            ),
            const Text(
              '.',
              style: TextStyle(
                fontSize: 34,
                fontWeight: FontWeight.w900,
                color: NanoColors.pink,
              ),
            ),
            const SizedBox(width: 12),
            const Padding(
              padding: EdgeInsets.only(bottom: 6),
              child: Text('audio generation',
                  style: TextStyle(color: NanoColors.textDim, fontSize: 13)),
            ),
            const Spacer(),
            _healthPill(),
          ],
        ),
        const SizedBox(height: 14),
        Row(
          children: [
            Expanded(
              child: TextField(
                controller: _serverCtl,
                style: const TextStyle(fontSize: 13, color: NanoColors.text),
                decoration: const InputDecoration(
                  hintText: 'server url',
                  prefixIcon:
                      Icon(Icons.dns_outlined, size: 16, color: NanoColors.textDim),
                ),
                onSubmitted: (_) => _checkHealth(),
              ),
            ),
            const SizedBox(width: 8),
            IconButton(
              onPressed: _healthLoading ? null : _checkHealth,
              icon: const Icon(Icons.refresh, color: NanoColors.pink),
              tooltip: 'check /health',
            ),
          ],
        ),
        if (_health != null) ...[
          const SizedBox(height: 6),
          _healthDetail(_health!),
        ],
      ],
    );
  }

  Widget _healthPill() {
    final (color, text) = _healthLoading
        ? (NanoColors.textDim, 'connecting')
        : _health != null
            ? (NanoColors.ok, 'online')
            : (NanoColors.error, 'offline');
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        border: Border.all(color: color),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 7,
            height: 7,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 6),
          Text(text, style: TextStyle(color: color, fontSize: 11)),
        ],
      ),
    );
  }

  Widget _healthDetail(HealthInfo h) {
    final params = '${(h.modelParams / 1e6).toStringAsFixed(0)}M';
    final bits = [
      h.device,
      'step ${h.ckptStep}',
      '$params params',
      h.textConditioning ? 'text-cond on' : 'text-cond off',
    ].join('  ·  ');
    return Text(bits,
        style: const TextStyle(color: NanoColors.textDim, fontSize: 11));
  }

  Widget _modeSelector() {
    return SegmentedRow<NanoMode>(
      options: NanoMode.values,
      selected: _mode,
      labelOf: (m) => m.label,
      onSelect: (m) => setState(() => _mode = m),
    );
  }

  Widget _conditioningCard() {
    return SectionCard(
      title: 'conditioning',
      children: [
        NanoTextField(
          label: 'tags / style prompt',
          controller: _promptCtl,
          maxLines: 2,
          hint: 'e.g. dark synthwave, driving bass, analog warmth',
          help: 'Genre / timbre / vibe. Encoded via CLAP at cross-attention '
              'position 0.',
        ),
        NanoTextField(
          label: 'lyrics',
          controller: _lyricsCtl,
          maxLines: 4,
          hint: 'the actual words to sing',
          help: 'Phonemized (g2p) into a token sequence for the lyric '
              'cross-attention. This is what makes the model sing words.',
        ),
        NanoTextField(
          label: 'negative prompt',
          controller: _negCtl,
          maxLines: 2,
          hint: 'what to steer away from',
          help: 'Classifier-free guidance pushes away from this prompt.',
        ),
        ToggleRow(
          label: 'sweeten prompt',
          value: _params.sweeten,
          help: 'Rewrite the tags prompt into LP-MusicCaps caption style to '
              'strengthen CLAP adherence. On by default.',
          onChanged: (v) => setState(() => _params.sweeten = v),
        ),
        const SizedBox(height: 10),
        _styleAudioRow(),
        if (_params.styleAudio != null)
          LabeledSlider(
            label: 'style weight',
            value: _params.styleWeight,
            min: 0,
            max: 1,
            help: 'Blend of style-audio embedding vs. tags embedding.',
            onChanged: (v) => setState(() => _params.styleWeight = v),
          ),
      ],
    );
  }

  Widget _styleAudioRow() {
    return _filePickRow(
      label: 'style audio (optional)',
      file: _params.styleAudio,
      onPick: () async {
        final f = await _pickAudio();
        if (f != null) setState(() => _params.styleAudio = f);
      },
      onClear: () => setState(() => _params.styleAudio = null),
    );
  }

  Widget _samplingCard() {
    return SectionCard(
      title: 'advanced',
      collapsible: true,
      initiallyExpanded: false,
      children: [
        LabeledSlider(
          label: 'creativity',
          value: _params.temperature,
          min: 0.1,
          max: 2.0,
          help: 'How adventurous the model is (overridden by the per-codebook field).',
          minLabel: 'safe',
          maxLabel: 'wild',
          defaultValue: GenParams.defaultTemperature,
          describe: (v) => v < 0.6
              ? 'safe & predictable'
              : v < 0.9
                  ? 'controlled'
                  : v < 1.15
                      ? 'balanced'
                      : v < 1.5
                          ? 'adventurous'
                          : 'wild / risky',
          onChanged: (v) => setState(() => _params.temperature = v),
        ),
        LabeledSlider(
          label: 'note probability',
          value: _params.topK.toDouble(),
          min: 0,
          max: 200,
          divisions: 200,
          fractionDigits: 0,
          help: 'How many of the most-likely notes stay on the table (0 = no limit).',
          minLabel: 'focused',
          maxLabel: 'anything goes',
          defaultValue: GenParams.defaultTopK.toDouble(),
          describe: (v) => v == 0
              ? 'no limit — all notes'
              : v < 30
                  ? 'very focused'
                  : v < 80
                      ? 'focused'
                      : v < 150
                          ? 'open'
                          : 'wide open',
          onChanged: (v) => setState(() => _params.topK = v.round()),
        ),
        LabeledSlider(
          label: 'note confidence',
          value: _params.topP,
          min: 0.0,
          max: 1.0,
          help: 'Keep just enough notes to cover this share of confidence (1.0 = no limit).',
          minLabel: 'tight',
          maxLabel: 'loose',
          defaultValue: GenParams.defaultTopP,
          describe: (v) => v >= 1.0
              ? 'no limit'
              : v < 0.85
                  ? 'tight'
                  : v < 0.97
                      ? 'balanced'
                      : 'loose',
          onChanged: (v) => setState(() => _params.topP = v),
        ),
        LabeledSlider(
          label: 'prompt adherence',
          value: _params.cfgScale,
          min: 1.0,
          max: 10.0,
          help: 'How strictly the model follows your prompt (tags + lyrics jointly).',
          minLabel: 'free',
          maxLabel: 'strict',
          defaultValue: GenParams.defaultCfgScale,
          describe: (v) => v < 2
              ? 'loose / free'
              : v < 4
                  ? 'balanced'
                  : v < 6
                      ? 'follows closely'
                      : v < 8
                          ? 'strict'
                          : 'forced / may distort',
          onChanged: (v) => setState(() => _params.cfgScale = v),
        ),
        LabeledSlider(
          label: 'lyric_cfg_scale',
          value: _params.lyricCfgScale,
          min: 0.0,
          max: 10.0,
          help: '0 = off. Pushes the lyric axis harder via composed guidance.',
          minLabel: 'off',
          maxLabel: 'max',
          defaultValue: GenParams.defaultLyricCfgScale,
          describe: (v) => v == 0
              ? 'off'
              : v < 3
                  ? 'gentle nudge'
                  : v < 6
                      ? 'clearer words'
                      : 'words forced',
          onChanged: (v) => setState(() => _params.lyricCfgScale = v),
        ),
        const SizedBox(height: 4),
        InkWell(
          onTap: () => setState(() => _showAdvanced = !_showAdvanced),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 6),
            child: Row(
              children: [
                Icon(
                  _showAdvanced ? Icons.expand_less : Icons.expand_more,
                  size: 18,
                  color: NanoColors.pink,
                ),
                const SizedBox(width: 4),
                const Text('per-codebook overrides',
                    style: TextStyle(color: NanoColors.pink, fontSize: 12)),
              ],
            ),
          ),
        ),
        if (_showAdvanced) ...[
          const SizedBox(height: 8),
          const Text(
            'comma-separated, length 9 — overrides the scalar above. '
            'a decreasing ladder usually sounds better.',
            style: TextStyle(color: NanoColors.textDim, fontSize: 11),
          ),
          const SizedBox(height: 12),
          NanoTextField(
            label: 'per_cb_temperature',
            controller: _perTempCtl,
            hint: '0.9,0.9,0.7,0.7,0.5,0.5,0.4,0.4,0.3',
          ),
          NanoTextField(
            label: 'per_cb_top_k',
            controller: _perTopKCtl,
            hint: '50,50,40,40,...',
          ),
          NanoTextField(
            label: 'per_cb_top_p',
            controller: _perTopPCtl,
            hint: '0.95,0.95,...',
          ),
        ],
      ],
    );
  }

  Widget _modeCard() {
    return switch (_mode) {
      NanoMode.generate => SectionCard(
          title: 'generate',
          children: [
            LabeledSlider(
              label: 'seconds',
              value: _params.seconds,
              min: 1,
              max: 95,
              fractionDigits: 0,
              suffix: 's',
              help: 'Single-shot generation length (max ~95s).',
              onChanged: (v) => setState(() => _params.seconds = v),
            ),
            const SizedBox(height: 12),
            ToggleRow(
              label: 'score CLAP adherence',
              value: _params.scoreClap,
              help: 'Return text<->audio CLAP similarity in a response header.',
              onChanged: (v) => setState(() => _params.scoreClap = v),
            ),
          ],
        ),
      NanoMode.continueAudio => SectionCard(
          title: 'continue',
          children: [
            _inputAudioRow(),
            const SizedBox(height: 8),
            LabeledSlider(
              label: 'add_seconds',
              value: _params.addSeconds,
              min: 1,
              max: 90,
              fractionDigits: 0,
              suffix: 's',
              help: 'How much new audio to generate after the prompt.',
              onChanged: (v) => setState(() => _params.addSeconds = v),
            ),
            LabeledSlider(
              label: 'prompt_seconds',
              value: _params.promptSeconds,
              min: 0,
              max: 30,
              fractionDigits: 0,
              suffix: 's',
              help: 'How much of the uploaded clip to use as the prompt '
                  '(0 = use all).',
              onChanged: (v) => setState(() => _params.promptSeconds = v),
            ),
          ],
        ),
      NanoMode.extend => SectionCard(
          title: 'extend',
          children: [
            _inputAudioRow(),
            const SizedBox(height: 8),
            LabeledSlider(
              label: 'add_seconds',
              value: _params.addSeconds,
              min: 1,
              max: 90,
              fractionDigits: 0,
              suffix: 's',
              help: 'How much new audio to append onto the end.',
              onChanged: (v) => setState(() => _params.addSeconds = v),
            ),
            LabeledSlider(
              label: 'overlap_seconds',
              value: _params.overlapSeconds,
              min: 1,
              max: 30,
              fractionDigits: 0,
              suffix: 's',
              help: 'Tail of the clip used as the prompt for the continuation.',
              onChanged: (v) => setState(() => _params.overlapSeconds = v),
            ),
          ],
        ),
    };
  }

  Widget _inputAudioRow() {
    return _filePickRow(
      label: 'input audio (required)',
      file: _params.inputAudio,
      onPick: () async {
        final f = await _pickAudio();
        if (f != null) setState(() => _params.inputAudio = f);
      },
      onClear: () => setState(() => _params.inputAudio = null),
    );
  }

  Widget _filePickRow({
    required String label,
    required AudioFile? file,
    required VoidCallback onPick,
    required VoidCallback onClear,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          OutlinedButton.icon(
            onPressed: onPick,
            icon: const Icon(Icons.upload_file, size: 16),
            label: const Text('choose'),
            style: OutlinedButton.styleFrom(
              foregroundColor: NanoColors.pink,
              side: const BorderSide(color: NanoColors.pink),
              shape:
                  RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              file?.name ?? label,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                color: file != null ? NanoColors.text : NanoColors.textDim,
                fontSize: 12,
              ),
            ),
          ),
          if (file != null)
            IconButton(
              onPressed: onClear,
              icon: const Icon(Icons.close, size: 16, color: NanoColors.textDim),
            ),
        ],
      ),
    );
  }

  Widget _runButton() {
    final needsAudio = _mode != NanoMode.generate;
    final disabled = _busy || (needsAudio && _params.inputAudio == null);
    return Padding(
      padding: const EdgeInsets.only(top: 4, bottom: 8),
      child: SizedBox(
        height: 52,
        child: ElevatedButton(
          onPressed: disabled ? null : _run,
          style: ElevatedButton.styleFrom(
            backgroundColor: NanoColors.pink,
            foregroundColor: Colors.black,
            disabledBackgroundColor: NanoColors.pinkDim,
            disabledForegroundColor: Colors.black54,
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
          ),
          child: _busy
              ? const SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(
                      strokeWidth: 2, color: Colors.black),
                )
              : Text(
                  _mode.label.toUpperCase(),
                  style: const TextStyle(
                    fontWeight: FontWeight.w900,
                    letterSpacing: 2,
                    fontSize: 15,
                  ),
                ),
        ),
      ),
    );
  }

  Widget _errorBox() {
    return Container(
      margin: const EdgeInsets.only(top: 12),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        border: Border.all(color: NanoColors.error),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.error_outline, color: NanoColors.error, size: 18),
          const SizedBox(width: 10),
          Expanded(
            child: Text(_error!,
                style: const TextStyle(color: NanoColors.error, fontSize: 12)),
          ),
        ],
      ),
    );
  }

  Widget _resultCard() {
    final r = _result!;
    final ext = r.mime.contains('wav') ? 'wav' : 'mp3';
    return SectionCard(
      title: 'output',
      children: [
        WaveformPlayer(
            key: ValueKey(_blob!.url), url: _blob!.url, bytes: r.bytes),
        const SizedBox(height: 14),
        Row(
          children: [
            OutlinedButton.icon(
              onPressed: () => _blob!.download('nano-${_mode.label}.$ext'),
              icon: const Icon(Icons.download, size: 16),
              label: Text('download .$ext'),
              style: OutlinedButton.styleFrom(
                foregroundColor: NanoColors.pink,
                side: const BorderSide(color: NanoColors.pink),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(4)),
              ),
            ),
            const Spacer(),
            if (r.clapScore != null)
              Text('CLAP ${r.clapScore!.toStringAsFixed(3)}',
                  style: const TextStyle(
                      color: NanoColors.pink,
                      fontWeight: FontWeight.bold,
                      fontSize: 12)),
          ],
        ),
        if (r.sweetened != null) ...[
          const SizedBox(height: 14),
          const _SubLabel('sweetened prompt'),
          const SizedBox(height: 4),
          Text(r.sweetened!,
              style: const TextStyle(
                  color: NanoColors.textDim,
                  fontSize: 12,
                  fontStyle: FontStyle.italic)),
        ],
      ],
    );
  }
}

class _SubLabel extends StatelessWidget {
  const _SubLabel(this.text);
  final String text;
  @override
  Widget build(BuildContext context) => Text(text,
      style: const TextStyle(color: NanoColors.text, fontSize: 13));
}
