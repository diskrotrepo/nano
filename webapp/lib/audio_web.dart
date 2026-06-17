import 'dart:async';
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

  /// Read this clip's duration (seconds) from a detached `<audio>` element's
  /// metadata. Returns 0 if it can't be determined.
  Future<double> duration() => audioDurationSeconds(url);

  void download(String filename) {
    final a = web.document.createElement('a') as web.HTMLAnchorElement
      ..href = url
      ..download = filename;
    web.document.body!.appendChild(a);
    a.click();
    a.remove();
  }
}

/// Download a streaming HTTP response (e.g. `/generate_stream`) to completion via
/// the Fetch API + ReadableStream, reporting bytes as they arrive. Unlike a
/// native `<audio src>`, this plays nicely with our live chunked stream (no
/// Content-Length / no Range support) AND keeps the connection actively reading,
/// so it slips under Modal's 150s web-endpoint wall — then the caller plays the
/// finished blob (which browsers play reliably; a live stream they will not).
Future<Uint8List> fetchAudioStream(
  String url, {
  void Function(int receivedBytes)? onProgress,
}) async {
  final resp = await web.window.fetch(url.toJS).toDart;
  if (resp.status != 200) {
    throw Exception('stream HTTP ${resp.status}');
  }
  final body = resp.body;
  if (body == null) {
    final buf = await resp.arrayBuffer().toDart;
    return buf.toDart.asUint8List();
  }
  final reader = body.getReader() as web.ReadableStreamDefaultReader;
  final chunks = <Uint8List>[];
  var received = 0;
  while (true) {
    final result = await reader.read().toDart;
    if (result.done) break;
    final value = result.value;
    if (value.isUndefinedOrNull) continue;
    final chunk = (value as JSUint8Array).toDart;
    chunks.add(chunk);
    received += chunk.length;
    onProgress?.call(received);
  }
  final out = Uint8List(received);
  var off = 0;
  for (final c in chunks) {
    out.setRange(off, off + c.length, c);
    off += c.length;
  }
  return out;
}

/// True if the browser can play MP3 through MediaSource Extensions (Chrome can;
/// Safari/Firefox vary). When false, callers fall back to [fetchAudioStream].
bool mseMp3Supported() {
  try {
    return web.MediaSource.isTypeSupported('audio/mpeg');
  } catch (_) {
    return false;
  }
}

/// Build a POST `RequestInit` carrying a multipart form (text [fields] plus one
/// uploaded file) — used to stream /extend and /cover, which need a file upload a
/// GET URL can't carry. Pass the result to [MseStream]'s `requestInit`.
web.RequestInit streamPostInit({
  required Map<String, String> fields,
  required String fileField,
  required Uint8List fileBytes,
  required String fileName,
}) {
  final fd = web.FormData();
  fields.forEach((k, v) => fd.append(k, v.toJS));
  final blob = web.Blob(
    [fileBytes.toJS].toJS,
    web.BlobPropertyBag(type: 'application/octet-stream'),
  );
  fd.append(fileField, blob, fileName);
  return web.RequestInit(method: 'POST', body: fd);
}

/// Progressive MP3 playback over MediaSource Extensions, as a self-contained
/// mini-player (a [ChangeNotifier]). Feeds a live `/generate_stream` response
/// into a SourceBuffer chunk-by-chunk, so playback can start before the clip is
/// finished — a native `<audio src>` can't play our live stream (no
/// Content-Length / Range). It owns its own detached `<audio>` element and starts
/// buffering on construction (attaching the MediaSource is what fires
/// `sourceopen`). [bufferedSeconds] reports how much is decodable so the UI can
/// gate play until enough is buffered; bytes are accumulated so the finished clip
/// can be downloaded / waveformed. Reading the stream to completion is also what
/// makes the server persist the finished clip to disk.
class MseStream extends ChangeNotifier {
  MseStream(String fetchUrl, {this.onPlay, web.RequestInit? requestInit}) {
    _fetchUrl = fetchUrl;
    _init = requestInit;
    _audio = web.document.createElement('audio') as web.HTMLAudioElement
      ..preload = 'auto';
    _objectUrl = web.URL.createObjectURL(_ms);
    _ms.addEventListener('sourceopen', _onSourceOpen.toJS);
    // Attaching the MediaSource to an element is what fires `sourceopen` and lets
    // buffering begin (before the user ever presses play).
    _audio.src = _objectUrl;
    _audio.addEventListener('timeupdate', ((web.Event _) => notifyListeners()).toJS);
    _audio.addEventListener('playing', _onPlaying.toJS);
    _audio.addEventListener('pause', _onPause.toJS);
    _audio.addEventListener('ended', _onPause.toJS);
  }

  final web.MediaSource _ms = web.MediaSource();
  late final web.HTMLAudioElement _audio;
  late final String _objectUrl;
  late final String _fetchUrl;
  web.RequestInit? _init; // POST request init (extend/cover upload), null = GET

  /// Invoked when this stream starts playing — lets the host stop other players
  /// so only one clip plays at a time.
  final void Function()? onPlay;

  web.SourceBuffer? _sb;
  final List<JSUint8Array> _queue = [];
  final List<Uint8List> _bytes = [];
  bool _streamDone = false;
  bool complete = false;
  String? error;
  int receivedBytes = 0;
  bool _playing = false;

  bool get playing => _playing;

  void _onPlaying(web.Event _) {
    _playing = true;
    notifyListeners();
  }

  void _onPause(web.Event _) {
    _playing = false;
    notifyListeners();
  }

  void togglePlay() {
    if (_playing) {
      _audio.pause();
    } else {
      onPlay?.call();
      _audio.play();
    }
  }

  void pause() => _audio.pause();

  /// Reference duration for the transport: the finished length once the stream
  /// ends (duration is Infinity until then), else how much is buffered — so you
  /// can scrub within what's streamed so far, mid-generation.
  double get _refDuration {
    final d = _audio.duration;
    if (d.isFinite && d > 0) return d;
    return bufferedSeconds;
  }

  /// Playback position as a fraction 0..1 of [_refDuration].
  double get positionFraction {
    final ref = _refDuration;
    if (ref <= 0) return 0;
    return (_audio.currentTime / ref).clamp(0.0, 1.0);
  }

  /// Scrub to [f] (0..1). Mid-stream this is a fraction of what's buffered; we
  /// never seek past the buffered frontier (seeking into a gap would stall).
  void seekFraction(double f) {
    final ref = _refDuration;
    if (ref <= 0) return;
    var t = f.clamp(0.0, 1.0) * ref;
    final maxT = bufferedSeconds > 0 ? bufferedSeconds : ref;
    if (t > maxT) t = maxT;
    _audio.currentTime = t;
    notifyListeners();
  }

  void _onSourceOpen(web.Event _) {
    if (_sb != null) return;
    try {
      final sb = _ms.addSourceBuffer('audio/mpeg');
      _sb = sb;
      sb.addEventListener('updateend', ((web.Event _) => _pump()).toJS);
      _startFetch();
    } catch (e) {
      error = '$e';
      notifyListeners();
    }
  }

  Future<void> _startFetch() async {
    try {
      final init = _init;
      final resp = await (init == null
              ? web.window.fetch(_fetchUrl.toJS)
              : web.window.fetch(_fetchUrl.toJS, init))
          .toDart;
      if (resp.status != 200) throw Exception('stream HTTP ${resp.status}');
      final reader = resp.body!.getReader() as web.ReadableStreamDefaultReader;
      while (true) {
        final r = await reader.read().toDart;
        if (r.done) break;
        final value = r.value;
        if (value.isUndefinedOrNull) continue;
        final arr = value as JSUint8Array;
        _bytes.add(arr.toDart);
        receivedBytes += arr.toDart.length;
        _queue.add(arr);
        _pump();
        notifyListeners();
      }
      _streamDone = true;
      _pump();
    } catch (e) {
      error = '$e';
      notifyListeners();
    }
  }

  void _pump() {
    final sb = _sb;
    if (sb == null || sb.updating) return;
    if (_queue.isNotEmpty) {
      final chunk = _queue.removeAt(0);
      try {
        sb.appendBuffer(chunk as JSObject);
      } catch (e) {
        error = '$e';
        notifyListeners();
      }
    } else if (_streamDone && _ms.readyState == 'open') {
      try {
        _ms.endOfStream();
      } catch (_) {}
      complete = true;
      notifyListeners();
    }
  }

  /// Decodable buffered duration (seconds) — the gate for enabling play.
  double get bufferedSeconds {
    final b = _sb?.buffered;
    if (b == null || b.length == 0) return 0;
    try {
      return b.end(b.length - 1).toDouble();
    } catch (_) {
      return 0;
    }
  }

  /// All bytes received so far, concatenated (for download / waveform on done).
  Uint8List allBytes() {
    final total = Uint8List(receivedBytes);
    var off = 0;
    for (final c in _bytes) {
      total.setRange(off, off + c.length, c);
      off += c.length;
    }
    return total;
  }

  @override
  void dispose() {
    try {
      _audio.pause();
      _audio.removeAttribute('src');
    } catch (_) {}
    try {
      web.URL.revokeObjectURL(_objectUrl);
    } catch (_) {}
    super.dispose();
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
Future<List<double>> decodeWaveformPeaks(Uint8List bytes, int buckets) async =>
    (await decodeWaveform(bytes, buckets)).peaks;

/// Decode `bytes` into `buckets` normalized peaks (0..1) **and** the clip's true
/// duration in seconds (read from the decoded buffer, so callers don't have to
/// trust a separately-measured length). Returns empty peaks / 0 duration if
/// decoding yields no samples.
Future<({List<double> peaks, double duration})> decodeWaveform(
    Uint8List bytes, int buckets) async {
  final ctx = web.AudioContext();
  try {
    // decodeAudioData *detaches* the input ArrayBuffer, so decoding the same
    // clip.bytes twice (mini-waveform + extend waveform) would fail the second
    // time. Hand it a fresh copy and leave the caller's bytes intact.
    final copy = Uint8List.fromList(bytes);
    final audioBuf = await ctx.decodeAudioData(copy.buffer.toJS).toDart;
    final duration = audioBuf.duration.toDouble();
    final channel = audioBuf.getChannelData(0).toDart; // Float32List
    final n = channel.length;
    if (n == 0) return (peaks: const <double>[], duration: duration);
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
    return (peaks: out, duration: duration);
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
                    painter: WaveformBarsPainter(peaks: _peaks, progress: progress),
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

/// Shared waveform bar painter, reused by [WaveformPlayer], the clip-card
/// scrubber, and the extend cut widget. Two display modes:
///   - playback: pass [progress] (0..1) — bars before the playhead are pink,
///     after are dim; a playhead line is drawn.
///   - cut: pass [cut] (0..1) — bars before the cut are pink ("kept"), after
///     are greyed ("discarded / regenerated"); a pink cut marker is drawn.
/// Either or both may be supplied; pass neither for a static dim waveform.
class WaveformBarsPainter extends CustomPainter {
  WaveformBarsPainter({this.peaks, this.progress, this.cut});
  final List<double>? peaks;
  final double? progress;
  final double? cut;

  @override
  void paint(Canvas canvas, Size size) {
    final mid = size.height / 2;
    final p = peaks;
    final progressX = progress == null ? null : size.width * progress!;
    final cutX = cut == null ? null : (size.width * cut!).clamp(0.0, size.width);

    if (p == null || p.isEmpty) {
      // Loading / undecodable: a dim centerline so the area isn't empty.
      canvas.drawLine(Offset(0, mid), Offset(size.width, mid),
          Paint()..color = NanoColors.pinkDim..strokeWidth = 2);
    } else {
      final played = Paint()..color = NanoColors.pink;
      final unplayed = Paint()..color = NanoColors.pinkDim;
      final discarded = Paint()..color = NanoColors.border;
      final n = p.length;
      final slot = size.width / n;
      final barW = (slot * 0.6).clamp(1.0, slot);
      for (var i = 0; i < n; i++) {
        final cx = i * slot + slot / 2;
        final h = (p[i] * size.height * 0.95).clamp(2.0, size.height);
        final rect = Rect.fromLTWH(cx - barW / 2, mid - h / 2, barW, h);
        final Paint paint;
        if (cutX != null) {
          paint = cx <= cutX ? played : discarded;
        } else if (progressX != null) {
          paint = cx <= progressX ? played : unplayed;
        } else {
          paint = unplayed;
        }
        canvas.drawRRect(
          RRect.fromRectAndRadius(rect, const Radius.circular(1)),
          paint,
        );
      }
    }

    // Playback playhead.
    if (progressX != null) {
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

    // Cut marker — where the extension begins.
    if (cutX != null) {
      final paint = Paint()
        ..color = NanoColors.pink
        ..strokeWidth = 2;
      canvas.drawLine(Offset(cutX, 0), Offset(cutX, size.height), paint);
      canvas.drawCircle(Offset(cutX, 0), 4, Paint()..color = NanoColors.pink);
      canvas.drawCircle(
          Offset(cutX, size.height), 4, Paint()..color = NanoColors.pink);
    }
  }

  @override
  bool shouldRepaint(WaveformBarsPainter old) =>
      old.progress != progress ||
      old.cut != cut ||
      !identical(old.peaks, peaks);
}

/// Read an audio clip's duration (seconds) from its blob URL via a detached
/// `<audio>` element's metadata. Returns 0 if it can't be determined.
Future<double> audioDurationSeconds(String url) {
  final c = Completer<double>();
  final el = web.document.createElement('audio') as web.HTMLAudioElement
    ..preload = 'metadata'
    ..src = url;
  void finish(double v) {
    if (!c.isCompleted) c.complete(v);
  }

  el.addEventListener('loadedmetadata', (web.Event _) {
    final d = el.duration;
    finish(d.isFinite && d > 0 ? d : 0);
  }.toJS);
  el.addEventListener('error', (web.Event _) {
    finish(0);
  }.toJS);
  return c.future;
}

/// One shared `<audio>` element backing the whole clip library, so only one
/// clip can ever play at a time: pressing play on a clip stops whatever was
/// playing. A [ChangeNotifier] so each clip card can reflect its own playing
/// state and the live playback progress (0..1).
class ClipPlayer extends ChangeNotifier {
  ClipPlayer() {
    _audio = web.document.createElement('audio') as web.HTMLAudioElement
      ..preload = 'auto';
    _audio.addEventListener('timeupdate', (web.Event _) {
      final d = _audio.duration;
      if (d.isFinite && d > 0) {
        _progress = _audio.currentTime / d;
        notifyListeners();
      }
    }.toJS);
    _audio.addEventListener('ended', (web.Event _) {
      _playing = false;
      _progress = 1.0;
      notifyListeners();
    }.toJS);
    // A seek requested before the (possibly just-swapped) source has metadata
    // can't set currentTime yet — apply the stashed fraction once it loads.
    _audio.addEventListener('loadedmetadata', (web.Event _) {
      final f = _pendingSeek;
      if (f == null) return;
      _pendingSeek = null;
      final d = _audio.duration;
      if (d.isFinite && d > 0) {
        _audio.currentTime = f * d;
        _progress = f;
        notifyListeners();
      }
    }.toJS);
  }

  late final web.HTMLAudioElement _audio;
  String? _currentId;
  String? _currentUrl;
  bool _playing = false;
  double _progress = 0;
  double? _pendingSeek;

  String? get currentId => _currentId;
  bool get isPlaying => _playing;
  double get progress => _progress;
  bool isCurrent(String id) => _currentId == id;

  /// Play/pause [id] (backed by [url]). Starting a different clip — OR the same
  /// clip with a new source (e.g. a streamed clip upgrading from its live stream
  /// URL to the gapless canonical blob once it arrives) — swaps the source and
  /// stops the previous one.
  void toggle(String id, String url) {
    if (_currentId == id && _currentUrl == url) {
      if (_playing) {
        _audio.pause();
        _playing = false;
      } else {
        if (_audio.ended || _progress >= 1.0) {
          _audio.currentTime = 0;
          _progress = 0;
        }
        _audio.play();
        _playing = true;
      }
      notifyListeners();
      return;
    }
    _audio.pause();
    _audio.src = url;
    _audio.currentTime = 0;
    _currentId = id;
    _currentUrl = url;
    _progress = 0;
    _playing = true;
    _audio.play();
    notifyListeners();
  }

  /// Seek clip [id] (backed by blob [url]) to [fraction] (0..1) of its length.
  /// If it isn't the loaded clip, load it (paused) and apply the seek once its
  /// metadata arrives. Drives the clip-card waveform scrubber.
  void seek(String id, String url, double fraction) {
    final f = fraction.clamp(0.0, 1.0);
    if (_currentId != id || _currentUrl != url) {
      _audio.pause();
      _audio.src = url;
      _currentId = id;
      _currentUrl = url;
      _playing = false;
      _progress = f;
      _pendingSeek = f; // applied on loadedmetadata
      notifyListeners();
      return;
    }
    final d = _audio.duration;
    if (d.isFinite && d > 0) {
      _audio.currentTime = f * d;
      _progress = f;
      notifyListeners();
    } else {
      _pendingSeek = f;
    }
  }

  /// Pause playback (without forgetting the loaded clip) — used to enforce
  /// one-at-a-time when a streaming clip starts.
  void pause() {
    if (_playing) {
      _audio.pause();
      _playing = false;
      notifyListeners();
    }
  }

  /// Stop and forget [id] if it's the one currently loaded (used when a clip is
  /// removed from the library).
  void stopIfCurrent(String id) {
    if (_currentId != id) return;
    _audio.pause();
    _audio.removeAttribute('src');
    _currentId = null;
    _currentUrl = null;
    _playing = false;
    _progress = 0;
    notifyListeners();
  }

  @override
  void dispose() {
    _audio.pause();
    _audio.removeAttribute('src');
    super.dispose();
  }
}
