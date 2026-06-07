import 'dart:js_interop';
import 'dart:typed_data';
import 'dart:ui_web' as ui_web;

import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

import 'theme.dart';

/// Wraps a freshly-created object URL for some audio bytes. Call [revoke] when
/// replacing it so we don't leak blob URLs across generations.
class AudioBlob {
  AudioBlob._(this.url, this.mime);
  final String url;
  final String mime;

  static AudioBlob fromBytes(Uint8List bytes, String mime) {
    final jsBytes = bytes.toJS;
    final blob = web.Blob(
      [jsBytes].toJS,
      web.BlobPropertyBag(type: mime),
    );
    final url = web.URL.createObjectURL(blob);
    return AudioBlob._(url, mime);
  }

  void revoke() => web.URL.revokeObjectURL(url);

  void download(String filename) {
    final a = web.document.createElement('a') as web.HTMLAnchorElement
      ..href = url
      ..download = filename;
    web.document.body!.appendChild(a);
    a.click();
    a.remove();
  }
}

/// A native HTML `<audio controls>` element bound to a blob URL. Registered
/// lazily per-URL so Flutter web can embed it via [HtmlElementView].
class BlobAudioPlayer extends StatefulWidget {
  const BlobAudioPlayer({super.key, required this.url});
  final String url;

  @override
  State<BlobAudioPlayer> createState() => _BlobAudioPlayerState();
}

class _BlobAudioPlayerState extends State<BlobAudioPlayer> {
  late final String _viewType;

  @override
  void initState() {
    super.initState();
    _viewType = 'nano-audio-${widget.url.hashCode}-${identityHashCode(this)}';
    ui_web.platformViewRegistry.registerViewFactory(_viewType, (int _) {
      final el = web.document.createElement('audio') as web.HTMLAudioElement
        ..src = widget.url
        ..controls = true
        ..style.width = '100%'
        ..style.height = '54px';
      return el;
    });
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(height: 54, child: HtmlElementView(viewType: _viewType));
  }
}

/// Decode `bytes` (mp3/wav) into `buckets` normalized peak amplitudes (0..1) for
/// drawing a waveform. Uses the browser's Web Audio decoder, so it handles any
/// format `<audio>` can play. Channel 0 only; max-abs per bucket; normalized so
/// quiet clips still fill the height. Returns `[]` if decoding yields no samples.
Future<List<double>> decodeWaveformPeaks(Uint8List bytes, int buckets) async {
  final ctx = web.AudioContext();
  try {
    final audioBuf = await ctx.decodeAudioData(bytes.buffer.toJS).toDart;
    final channel = audioBuf.getChannelData(0).toDart; // Float32List
    final n = channel.length;
    if (n == 0) return const [];
    final out = List<double>.filled(buckets, 0);
    final per = (n / buckets).ceil();
    var peak = 0.0;
    for (var b = 0; b < buckets; b++) {
      final start = b * per;
      if (start >= n) break;
      final end = (start + per > n) ? n : start + per;
      var maxAbs = 0.0;
      for (var i = start; i < end; i++) {
        final v = channel[i].abs();
        if (v > maxAbs) maxAbs = v;
      }
      out[b] = maxAbs;
      if (maxAbs > peak) peak = maxAbs;
    }
    if (peak > 0) {
      for (var b = 0; b < buckets; b++) {
        out[b] = out[b] / peak;
      }
    }
    return out;
  } finally {
    ctx.close();
  }
}

/// Renders generated audio as a waveform with a custom transport — decoded peak
/// bars that fill pink as playback advances, a draggable playhead (click/drag to
/// scrub), and play / pause / stop / loop buttons. No native `<audio controls>`:
/// the `<audio>` element is created detached and driven purely by these controls.
/// Falls back to a flat baseline while peaks decode (or if decoding fails), so
/// playback always works regardless.
class WaveformPlayer extends StatefulWidget {
  const WaveformPlayer({super.key, required this.url, required this.bytes});
  final String url;
  final Uint8List bytes;

  @override
  State<WaveformPlayer> createState() => _WaveformPlayerState();
}

class _WaveformPlayerState extends State<WaveformPlayer> {
  static const int _buckets = 320;

  final ValueNotifier<double> _progress = ValueNotifier<double>(0);
  late final web.HTMLAudioElement _audio;
  List<double>? _peaks;
  bool _playing = false;
  bool _looping = false;

  @override
  void initState() {
    super.initState();
    // Detached element: never added to the DOM, so no browser chrome shows —
    // it's just the playback engine for our custom controls.
    _audio = web.document.createElement('audio') as web.HTMLAudioElement
      ..src = widget.url
      ..preload = 'auto';
    _audio.addEventListener('timeupdate', (web.Event _) {
      final d = _audio.duration;
      if (d.isFinite && d > 0) _progress.value = _audio.currentTime / d;
    }.toJS);
    _audio.addEventListener('ended', (web.Event _) {
      if (!_audio.loop && mounted) {
        _progress.value = 1.0;
        setState(() => _playing = false);
      }
    }.toJS);
    _decode();
  }

  Future<void> _decode() async {
    List<double> peaks;
    try {
      peaks = await decodeWaveformPeaks(widget.bytes, _buckets);
    } catch (_) {
      peaks = const []; // decoding failed — keep the controls, skip the bars
    }
    if (mounted) setState(() => _peaks = peaks);
  }

  void _play() {
    _audio.play();
    setState(() => _playing = true);
  }

  void _pause() {
    _audio.pause();
    setState(() => _playing = false);
  }

  void _stop() {
    _audio.pause();
    _audio.currentTime = 0;
    _progress.value = 0;
    setState(() => _playing = false);
  }

  void _toggleLoop() {
    setState(() {
      _looping = !_looping;
      _audio.loop = _looping;
    });
  }

  void _seekToFraction(double fraction) {
    final d = _audio.duration;
    if (!d.isFinite || d <= 0) return;
    final f = fraction.clamp(0.0, 1.0);
    _audio.currentTime = f * d;
    _progress.value = f;
  }

  @override
  void dispose() {
    _audio.pause();
    _audio.removeAttribute('src');
    _progress.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SizedBox(
          height: 72,
          child: LayoutBuilder(
            builder: (context, constraints) {
              final w = constraints.maxWidth;
              return GestureDetector(
                onTapDown: (d) => _seekToFraction(d.localPosition.dx / w),
                onHorizontalDragStart: (d) =>
                    _seekToFraction(d.localPosition.dx / w),
                onHorizontalDragUpdate: (d) =>
                    _seekToFraction(d.localPosition.dx / w),
                child: ValueListenableBuilder<double>(
                  valueListenable: _progress,
                  builder: (context, progress, _) => CustomPaint(
                    size: Size.infinite,
                    painter: _WaveformPainter(_peaks, progress),
                  ),
                ),
              );
            },
          ),
        ),
        const SizedBox(height: 6),
        Row(
          children: [
            _TransportButton(
              icon: _playing ? Icons.pause : Icons.play_arrow,
              tooltip: _playing ? 'pause' : 'play',
              onPressed: _playing ? _pause : _play,
            ),
            _TransportButton(
              icon: Icons.stop,
              tooltip: 'stop',
              onPressed: _stop,
            ),
            _TransportButton(
              icon: Icons.loop,
              tooltip: _looping ? 'loop on' : 'loop off',
              active: _looping,
              onPressed: _toggleLoop,
            ),
          ],
        ),
      ],
    );
  }
}

class _TransportButton extends StatelessWidget {
  const _TransportButton({
    required this.icon,
    required this.tooltip,
    required this.onPressed,
    this.active = false,
  });
  final IconData icon;
  final String tooltip;
  final VoidCallback onPressed;
  final bool active;

  @override
  Widget build(BuildContext context) {
    return IconButton(
      onPressed: onPressed,
      tooltip: tooltip,
      iconSize: 20,
      color: active ? NanoColors.pink : NanoColors.text,
      icon: Icon(icon),
    );
  }
}

class _WaveformPainter extends CustomPainter {
  _WaveformPainter(this.peaks, this.progress);
  final List<double>? peaks;
  final double progress;

  @override
  void paint(Canvas canvas, Size size) {
    final mid = size.height / 2;
    final p = peaks;
    final progressX = size.width * progress;

    if (p == null || p.isEmpty) {
      // Loading / undecodable: a dim centerline so the area isn't empty.
      canvas.drawLine(Offset(0, mid), Offset(size.width, mid),
          Paint()..color = NanoColors.pinkDim..strokeWidth = 2);
    } else {
      final played = Paint()..color = NanoColors.pink;
      final unplayed = Paint()..color = NanoColors.pinkDim;
      final n = p.length;
      final slot = size.width / n;
      final barW = (slot * 0.6).clamp(1.0, slot);
      for (var i = 0; i < n; i++) {
        final cx = i * slot + slot / 2;
        final h = (p[i] * size.height * 0.95).clamp(2.0, size.height);
        final rect = Rect.fromLTWH(cx - barW / 2, mid - h / 2, barW, h);
        canvas.drawRRect(
          RRect.fromRectAndRadius(rect, const Radius.circular(1)),
          cx <= progressX ? played : unplayed,
        );
      }
    }

    // Draggable playhead.
    canvas.drawLine(
      Offset(progressX, 0),
      Offset(progressX, size.height),
      Paint()
        ..color = NanoColors.text
        ..strokeWidth = 1.5,
    );
    canvas.drawCircle(
        Offset(progressX, 0), 3, Paint()..color = NanoColors.text);
  }

  @override
  bool shouldRepaint(_WaveformPainter old) =>
      old.progress != progress || !identical(old.peaks, peaks);
}
