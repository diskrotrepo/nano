import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

import 'api.dart';
import 'audio_web.dart';
import 'theme.dart';
import 'widgets.dart';

/// Height of the fixed top bar the left/right panels scroll behind.
const double _kTopBarHeight = 60;

void main() => runApp(const NanoApp());

class NanoApp extends StatelessWidget {
  const NanoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'diskrot///nano',
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

  /// Mode is derived from whether a source clip has been dropped onto the create
  /// area (sets `inputAudio`): no source → generate; source present → whichever
  /// transform the user picked, [_sourceMode] (extend or cover).
  NanoMode get _mode =>
      _params.inputAudio != null ? _sourceMode : NanoMode.generate;

  /// Which transform a dropped source drives. extend (continue the clip) or
  /// cover (re-render its melody in the prompt's timbre). Picked via the toggle
  /// that appears once a clip is dropped.
  NanoMode _sourceMode = NanoMode.extend;

  /// The library clip dropped as the source (for its name + waveform). Used by
  /// both extend and cover.
  GenClip? _sourceClip;

  final _serverCtl =
      TextEditingController(text: defaultServerUrl(web.window.location.hostname));
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

  final List<GenClip> _clips = [];
  final ClipPlayer _player = ClipPlayer();
  int _clipSeq = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _checkHealth());
  }

  @override
  void dispose() {
    _player.dispose();
    for (final clip in _clips) {
      clip.blob?.revoke();
    }
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

    final isStem = _mode == NanoMode.stem;
    final clip = GenClip(
      id: _clipSeq++,
      mode: _mode,
      prompt: isStem
          ? 'keep ${kStemNames.where(_params.stemKeep.contains).join(', ')}'
          : _promptCtl.text.trim(),
      lyrics: isStem ? '' : _lyricsCtl.text.trim(),
    );
    setState(() {
      _busy = true;
      _error = null;
      _clips.insert(0, clip);
    });
    try {
      final result = await _api.run(_mode, _params);
      final blob = AudioBlob.fromBytes(result.bytes, result.mime);
      final dur = await blob.duration();
      if (!mounted) {
        blob.revoke();
        return;
      }
      setState(() {
        clip
          ..status = ClipStatus.ready
          ..bytes = result.bytes
          ..mime = result.mime
          ..blob = blob
          ..durationSeconds = dur
          ..clapScore = result.clapScore
          ..sweetened = result.sweetened;
      });
    } catch (e) {
      if (mounted) {
        setState(() {
          clip
            ..status = ClipStatus.error
            ..error = '$e';
          _error = '$e';
        });
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        bottom: false,
        // The left + right panels fill the whole area and scroll *behind* a
        // fixed top bar (overlaid via the Stack). Each scroll view starts with
        // top padding equal to the bar height so nothing hides under it at rest.
        child: Stack(
          children: [
            Positioned.fill(child: _body()),
            Positioned(top: 0, left: 0, right: 0, child: _topBar()),
          ],
        ),
      ),
    );
  }

  Widget _body() {
    // Panels scroll under the bar, so the first item sits just below it.
    const topInset = _kTopBarHeight + 8;
    const pad = EdgeInsets.fromLTRB(20, topInset, 20, 48);
    return LayoutBuilder(
      builder: (context, constraints) {
        final wide = constraints.maxWidth >= 900;
        if (wide) {
          return Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              SizedBox(
                width: 560,
                child: _controlsArea(_controlsChildren(), pad),
              ),
              const VerticalDivider(width: 1, color: NanoColors.border),
              Expanded(
                child: ListView(padding: pad, children: _libraryChildren()),
              ),
            ],
          );
        }
        return _controlsArea([
          ..._controlsChildren(),
          const SizedBox(height: 8),
          const Divider(height: 1, color: NanoColors.border),
          const SizedBox(height: 16),
          ..._libraryChildren(),
        ], pad);
      },
    );
  }

  /// The create/controls column, made a drop target: dragging a ready library
  /// clip over it shows a "drop a track" overlay; dropping sets it as the source
  /// for extend / cover.
  Widget _controlsArea(List<Widget> children, EdgeInsets padding) {
    return DragTarget<GenClip>(
      onWillAcceptWithDetails: (d) =>
          d.data.status == ClipStatus.ready && d.data.bytes != null,
      onAcceptWithDetails: (d) => _acceptSource(d.data),
      builder: (context, candidate, rejected) {
        return Stack(
          children: [
            ListView(padding: padding, children: children),
            if (candidate.isNotEmpty)
              const Positioned.fill(
                child: Padding(
                  padding: EdgeInsets.all(12),
                  child: DropOverlay(label: 'drop to extend, cover or split stems'),
                ),
              ),
          ],
        );
      },
    );
  }

  /// Fixed full-width bar: branding + "audio generation" tagline on the left,
  /// the online/offline health pill on the right. The panels scroll behind it.
  Widget _topBar() {
    return Container(
      height: _kTopBarHeight,
      padding: const EdgeInsets.symmetric(horizontal: 20),
      decoration: const BoxDecoration(
        color: NanoColors.surface,
        border: Border(bottom: BorderSide(color: NanoColors.border)),
      ),
      child: Row(
        children: [
          const Text.rich(
            TextSpan(
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.w900,
                letterSpacing: -0.5,
                color: NanoColors.text,
              ),
              children: [
                TextSpan(text: 'diskrot'),
                WidgetSpan(child: SizedBox(width: 3)),
                TextSpan(text: '///', style: TextStyle(color: NanoColors.pink)),
                WidgetSpan(child: SizedBox(width: 3)),
                TextSpan(text: 'nano'),
              ],
            ),
            semanticsLabel: 'diskrot///nano',
          ),
          const SizedBox(width: 12),
          const Expanded(
            child: Text(
              'audio generation',
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(color: NanoColors.textDim, fontSize: 13),
            ),
          ),
          const SizedBox(width: 12),
          _healthPill(),
        ],
      ),
    );
  }

  List<Widget> _controlsChildren() {
    // Stem separation is pure DSP — no prompt/lyrics/sampling apply, so those
    // cards are hidden and the panel collapses to just the stem picker.
    final isStem = _mode == NanoMode.stem;
    return [
      _header(),
      const SizedBox(height: 18),
      if (!isStem) _conditioningCard(),
      if (!isStem) _samplingCard(),
      _modeCard(),
      _runButton(),
      if (_error != null) _errorBox(),
    ];
  }

  List<Widget> _libraryChildren() {
    return [
      Row(
        children: [
          const Text(
            'LIBRARY',
            style: TextStyle(
              color: NanoColors.pink,
              fontSize: 11,
              letterSpacing: 2,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(width: 8),
          Text(
            '${_clips.length}',
            style: const TextStyle(color: NanoColors.textDim, fontSize: 11),
          ),
        ],
      ),
      const SizedBox(height: 14),
      if (_clips.isEmpty)
        const Padding(
          padding: EdgeInsets.only(top: 48),
          child: Center(
            child: Text(
              'generated clips will appear here',
              style: TextStyle(color: NanoColors.textDim, fontSize: 13),
            ),
          ),
        )
      else
        for (final clip in _clips) _clipCardFor(clip),
    ];
  }

  Widget _clipCardFor(GenClip clip) {
    return ClipCard(
      key: ValueKey(clip.id),
      clip: clip,
      player: _player,
      onToggle: () => _player.toggle(clip.id.toString(), clip.blob!.url),
      onDownload: () {
        final ext = (clip.mime ?? '').contains('wav') ? 'wav' : 'mp3';
        clip.blob?.download('nano-${clip.mode.label}-${clip.id}.$ext');
      },
      onDelete: () => _deleteClip(clip),
    );
  }

  void _deleteClip(GenClip clip) {
    _player.stopIfCurrent(clip.id.toString());
    clip.blob?.revoke();
    setState(() => _clips.remove(clip));
  }

  Widget _header() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
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

  /// Set the dropped library clip as the source — flips the derived mode to the
  /// active transform (extend/cover) and seeds the extend cut marker at the tail.
  void _acceptSource(GenClip clip) {
    if (clip.status != ClipStatus.ready || clip.bytes == null) return;
    final ext = (clip.mime ?? '').contains('wav') ? 'wav' : 'mp3';
    setState(() {
      _sourceClip = clip;
      _params.inputAudio = AudioFile('nano-clip-${clip.id}.$ext', clip.bytes!);
      _params.fromSeconds = -1.0; // default: append at the tail
    });
  }

  /// Upload an external audio file as the source (e.g. a hum to cover) — same
  /// effect as dropping a library clip, but there's no GenClip behind it, so the
  /// extend cut-marker waveform is skipped (extend just appends at the tail).
  Future<void> _uploadSource() async {
    final f = await _pickAudio();
    if (f == null) return;
    setState(() {
      _sourceClip = null;
      _params.inputAudio = f;
      _params.fromSeconds = -1.0;
    });
  }

  /// Clear the source — back to generate.
  void _clearSource() {
    setState(() {
      _sourceClip = null;
      _params.inputAudio = null;
      _params.fromSeconds = -1.0;
    });
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
        const SizedBox(height: 10),
        _genderSelector(),
        NanoTextField(
          label: 'negative prompt',
          controller: _negCtl,
          maxLines: 2,
          hint: 'what to steer away from',
          help: 'Classifier-free guidance pushes away from this prompt.',
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
        ToggleRow(
          label: 'sweeten prompt',
          value: _params.sweeten,
          help: 'Rewrite the tags prompt into LP-MusicCaps caption style to '
              'strengthen CLAP adherence. On by default.',
          onChanged: (v) => setState(() => _params.sweeten = v),
        ),
        if (_mode == NanoMode.generate)
          ToggleRow(
            label: 'score CLAP adherence',
            value: _params.scoreClap,
            help: 'Return text<->audio CLAP similarity in a response header.',
            onChanged: (v) => setState(() => _params.scoreClap = v),
          ),
        const SizedBox(height: 10),
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
    if (_mode == NanoMode.generate) {
      return SectionCard(
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
          const SizedBox(height: 10),
          const Divider(height: 1, color: NanoColors.border),
          const SizedBox(height: 8),
          // Bring in a source track to transform: drag a library clip onto this
          // panel, or upload a file (e.g. a hum to cover). Either flips to the
          // extend/cover controls.
          _filePickRow(
            label: 'or drop / upload audio to extend, cover or split stems',
            file: null,
            onPick: _uploadSource,
            onClear: () {},
          ),
        ],
      );
    }
    // A source clip is dropped — pick the transform (extend or cover) with the
    // toggle, then show that transform's controls.
    return SectionCard(
      title: _mode.label,
      children: [
        _sourceClipRow(),
        const SizedBox(height: 12),
        _transformToggle(),
        const SizedBox(height: 14),
        ...switch (_mode) {
          NanoMode.cover => _coverControls(),
          NanoMode.stem => _stemControls(),
          _ => _extendControls(),
        },
      ],
    );
  }

  /// auto | male | female segmented selector for vocal gender. Sets
  /// [GenParams.gender] (''/'male'/'female'); the server injects the matching
  /// [male]/[female] marker into the lyric stream — it takes effect even with no
  /// lyrics (a gender-only header stream), so you can steer an instrumental
  /// generation's vocal toward male/female.
  Widget _genderSelector() {
    Widget seg(String value, String label, IconData icon) {
      final selected = _params.gender == value;
      return Expanded(
        child: InkWell(
          onTap: () => setState(() => _params.gender = value),
          child: Container(
            padding: const EdgeInsets.symmetric(vertical: 8),
            decoration: BoxDecoration(
              color: selected ? NanoColors.pink : Colors.transparent,
              border: Border.all(
                  color: selected ? NanoColors.pink : NanoColors.border),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(icon,
                    size: 13,
                    color: selected ? Colors.black : NanoColors.textDim),
                const SizedBox(width: 6),
                Text(
                  label,
                  style: TextStyle(
                    color: selected ? Colors.black : NanoColors.textDim,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(bottom: 6),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('vocal gender',
                  style: TextStyle(color: NanoColors.text, fontSize: 13)),
              const SizedBox(width: 6),
              Tooltip(
                message:
                    "Steers the lyric stream's gender marker. auto leaves it "
                    "unset (<unknown_gender>); male/female take effect even "
                    "with no lyrics. Needs a lyric-conditioned checkpoint.",
                waitDuration: const Duration(milliseconds: 300),
                textStyle: const TextStyle(color: Colors.black, fontSize: 12),
                decoration: BoxDecoration(
                  color: NanoColors.pink,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: const Icon(Icons.info_outline,
                    size: 13, color: NanoColors.textDim),
              ),
            ],
          ),
        ),
        Row(
          children: [
            seg('', 'auto', Icons.help_outline),
            const SizedBox(width: 8),
            seg('male', 'male', Icons.male),
            const SizedBox(width: 8),
            seg('female', 'female', Icons.female),
          ],
        ),
      ],
    );
  }

  /// extend | cover segmented toggle, shown once a source clip is dropped.
  Widget _transformToggle() {
    Widget seg(NanoMode m, IconData icon) {
      final selected = _sourceMode == m;
      return Expanded(
        child: InkWell(
          onTap: () => setState(() => _sourceMode = m),
          child: Container(
            padding: const EdgeInsets.symmetric(vertical: 9),
            decoration: BoxDecoration(
              color: selected ? NanoColors.pink : Colors.transparent,
              border: Border.all(
                  color: selected ? NanoColors.pink : NanoColors.border),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(icon,
                    size: 14,
                    color: selected ? Colors.black : NanoColors.textDim),
                const SizedBox(width: 6),
                Text(
                  m.label,
                  style: TextStyle(
                    color: selected ? Colors.black : NanoColors.textDim,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
        ),
      );
    }

    return Row(
      children: [
        seg(NanoMode.extend, Icons.east),
        const SizedBox(width: 8),
        seg(NanoMode.cover, Icons.brush),
        const SizedBox(width: 8),
        seg(NanoMode.stem, Icons.layers),
      ],
    );
  }

  List<Widget> _extendControls() {
    return [
      if (_sourceClip?.bytes != null)
        ExtendWaveform(
          bytes: _sourceClip!.bytes!,
          durationSeconds: _sourceClip!.durationSeconds ?? 0,
          fromSeconds: _params.fromSeconds,
          onChanged: (v) => setState(() => _params.fromSeconds = v),
        ),
      const SizedBox(height: 12),
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
    ];
  }

  List<Widget> _coverControls() {
    return [
      const Padding(
        padding: EdgeInsets.only(bottom: 14),
        child: Text(
          "Re-renders the dropped clip's melody in your prompt's timbre "
          '(e.g. hum → solo violin). The source audio is not kept — only its '
          'melodic contour conditions the output. The output length matches the '
          'source. Needs a melody-trained checkpoint.',
          style:
              TextStyle(color: NanoColors.textDim, fontSize: 11, height: 1.4),
        ),
      ),
      LabeledSlider(
        label: 'melody adherence',
        value: _params.melodyCfgScale,
        min: 0.0,
        max: 10.0,
        help: '0 = guided jointly with the prompt by prompt adherence. Higher '
            'pushes the cover to track your melody more tightly (its own '
            'guidance scale).',
        minLabel: 'off',
        maxLabel: 'max',
        defaultValue: GenParams.defaultMelodyCfgScale,
        describe: (v) => v == 0
            ? 'off (joint)'
            : v < 3
                ? 'gentle'
                : v < 6
                    ? 'tracks closely'
                    : 'forced',
        onChanged: (v) => setState(() => _params.melodyCfgScale = v),
      ),
    ];
  }

  List<Widget> _stemControls() {
    Widget stemChip(String stem) {
      final kept = _params.stemKeep.contains(stem);
      return InkWell(
        onTap: () => setState(() {
          if (kept) {
            _params.stemKeep.remove(stem);
          } else {
            _params.stemKeep.add(stem);
          }
        }),
        borderRadius: BorderRadius.circular(4),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
          decoration: BoxDecoration(
            color: kept ? NanoColors.pink : Colors.transparent,
            border:
                Border.all(color: kept ? NanoColors.pink : NanoColors.border),
            borderRadius: BorderRadius.circular(4),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(kept ? Icons.check : Icons.add,
                  size: 13, color: kept ? Colors.black : NanoColors.textDim),
              const SizedBox(width: 6),
              Text(stem,
                  style: TextStyle(
                    color: kept ? Colors.black : NanoColors.textDim,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  )),
            ],
          ),
        ),
      );
    }

    return [
      const Padding(
        padding: EdgeInsets.only(bottom: 14),
        child: Text(
          'Splits the dropped clip into stems with Demucs and keeps the ones you '
          'pick — the rest are dropped. No model is involved (works with any '
          'checkpoint), but the server needs the demucs package installed.',
          style:
              TextStyle(color: NanoColors.textDim, fontSize: 11, height: 1.4),
        ),
      ),
      const Text('keep stems',
          style: TextStyle(color: NanoColors.text, fontSize: 13)),
      const SizedBox(height: 8),
      Wrap(
        spacing: 8,
        runSpacing: 8,
        children: [for (final s in kStemNames) stemChip(s)],
      ),
      const SizedBox(height: 12),
      Wrap(
        spacing: 8,
        runSpacing: 8,
        children: [
          _stemPreset('instrumental', const {'drums', 'bass', 'other'}),
          _stemPreset('a-cappella', const {'vocals'}),
          _stemPreset('all', const {'drums', 'bass', 'other', 'vocals'}),
        ],
      ),
      if (_params.stemKeep.isEmpty) ...[
        const SizedBox(height: 10),
        const Text('select at least one stem to keep',
            style: TextStyle(color: NanoColors.error, fontSize: 11)),
      ],
    ];
  }

  /// Quick-set button: replaces the kept-stem set with a preset (instrumental /
  /// a-cappella / all). A fresh copy so the const preset set isn't mutated.
  Widget _stemPreset(String label, Set<String> keep) {
    return OutlinedButton(
      onPressed: () => setState(() => _params.stemKeep = {...keep}),
      style: OutlinedButton.styleFrom(
        foregroundColor: NanoColors.pink,
        side: const BorderSide(color: NanoColors.border),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
      ),
      child: Text(label, style: const TextStyle(fontSize: 11)),
    );
  }

  Widget _sourceClipRow() {
    final name = _sourceClip?.prompt.isNotEmpty == true
        ? _sourceClip!.prompt
        : (_params.inputAudio?.name ?? 'dropped track');
    return Row(
      children: [
        const Icon(Icons.link, size: 16, color: NanoColors.pink),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            name,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(color: NanoColors.text, fontSize: 12),
          ),
        ),
        IconButton(
          onPressed: _clearSource,
          icon: const Icon(Icons.close, size: 16, color: NanoColors.textDim),
          tooltip: 'clear — back to generate',
          visualDensity: VisualDensity.compact,
        ),
      ],
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
    final disabled = _busy ||
        (needsAudio && _params.inputAudio == null) ||
        (_mode == NanoMode.stem && _params.stemKeep.isEmpty);
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

}

/// One generated clip in the library: its request (mode + prompt + lyrics) plus,
/// once the request resolves, the audio bytes/blob, measured length, and any
/// CLAP score / sweetened prompt the server returned.
enum ClipStatus { generating, ready, error }

class GenClip {
  GenClip({
    required this.id,
    required this.mode,
    required this.prompt,
    required this.lyrics,
  });

  final int id;
  final NanoMode mode;
  final String prompt;
  final String lyrics;

  ClipStatus status = ClipStatus.generating;
  Uint8List? bytes;
  String? mime;
  AudioBlob? blob;
  double? durationSeconds;
  double? clapScore;
  String? sweetened;
  String? error;
}

String _fmtLen(double? s) {
  if (s == null || s <= 0) return '—';
  if (s >= 60) {
    final m = s ~/ 60;
    final sec = (s % 60).round().toString().padLeft(2, '0');
    return '$m:$sec';
  }
  return '${s.toStringAsFixed(1)}s';
}

/// A library row: play/pause (driven by the shared [ClipPlayer] so only one
/// clip plays at a time), the prompt, the clip length, a live playback bar, and
/// download / remove actions. Shows a spinner while generating and the error if
/// the request failed.
class ClipCard extends StatelessWidget {
  const ClipCard({
    super.key,
    required this.clip,
    required this.player,
    required this.onToggle,
    required this.onDownload,
    required this.onDelete,
  });

  final GenClip clip;
  final ClipPlayer player;
  final VoidCallback onToggle;
  final VoidCallback onDownload;
  final VoidCallback onDelete;

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: player,
      builder: (context, _) {
        final ready = clip.status == ClipStatus.ready;
        final isCurrent = player.isCurrent(clip.id.toString());
        final isPlaying = isCurrent && player.isPlaying;

        return Container(
          margin: const EdgeInsets.only(bottom: 12),
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: NanoColors.surface,
            border: Border.all(
                color: isCurrent ? NanoColors.pink : NanoColors.border),
            borderRadius: BorderRadius.circular(6),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _leading(isPlaying),
                  const SizedBox(width: 12),
                  Expanded(child: _body(ready)),
                  const SizedBox(width: 8),
                  _lengthChip(),
                ],
              ),
              if (ready) ...[
                const SizedBox(height: 10),
                ClipWaveform(
                  id: clip.id.toString(),
                  url: clip.blob!.url,
                  bytes: clip.bytes!,
                  player: player,
                ),
              ],
              if (ready && clip.sweetened != null) ...[
                const SizedBox(height: 8),
                Text(
                  clip.sweetened!,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    color: NanoColors.textDim,
                    fontSize: 11,
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ],
              const SizedBox(height: 4),
              _footer(ready),
            ],
          ),
        );
      },
    );
  }

  Widget _leading(bool isPlaying) {
    if (clip.status == ClipStatus.generating) {
      return const SizedBox(
        width: 38,
        height: 38,
        child: Padding(
          padding: EdgeInsets.all(9),
          child: CircularProgressIndicator(
              strokeWidth: 2, color: NanoColors.pink),
        ),
      );
    }
    if (clip.status == ClipStatus.error) {
      return Container(
        width: 38,
        height: 38,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(color: NanoColors.error),
        ),
        child: const Icon(Icons.error_outline,
            color: NanoColors.error, size: 20),
      );
    }
    return InkWell(
      onTap: onToggle,
      customBorder: const CircleBorder(),
      child: Container(
        width: 38,
        height: 38,
        alignment: Alignment.center,
        decoration: const BoxDecoration(
            shape: BoxShape.circle, color: NanoColors.pink),
        child: Icon(isPlaying ? Icons.pause : Icons.play_arrow,
            color: Colors.black, size: 22),
      ),
    );
  }

  Widget _body(bool ready) {
    final primary = switch (clip.status) {
      ClipStatus.generating => 'generating…',
      ClipStatus.error => clip.error ?? 'failed',
      ClipStatus.ready => clip.prompt.isEmpty ? '(no prompt)' : clip.prompt,
    };
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          primary,
          maxLines: 2,
          overflow: TextOverflow.ellipsis,
          style: TextStyle(
            color: clip.status == ClipStatus.error
                ? NanoColors.error
                : NanoColors.text,
            fontSize: 13,
            height: 1.25,
          ),
        ),
        if (ready && clip.lyrics.isNotEmpty) ...[
          const SizedBox(height: 4),
          Row(
            children: [
              const Icon(Icons.music_note, size: 12, color: NanoColors.textDim),
              const SizedBox(width: 4),
              Expanded(
                child: Text(
                  clip.lyrics,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                      color: NanoColors.textDim, fontSize: 11),
                ),
              ),
            ],
          ),
        ],
      ],
    );
  }

  Widget _lengthChip() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: NanoColors.surfaceAlt,
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        _fmtLen(clip.durationSeconds),
        style: const TextStyle(
            color: NanoColors.text, fontSize: 11, fontWeight: FontWeight.bold),
      ),
    );
  }

  Widget _footer(bool ready) {
    return Row(
      children: [
        if (ready) ...[
          _dragHandle(),
          const SizedBox(width: 10),
        ],
        Text(clip.mode.label,
            style: const TextStyle(color: NanoColors.textDim, fontSize: 11)),
        if (clip.clapScore != null) ...[
          const Text('  ·  ',
              style: TextStyle(color: NanoColors.textDim, fontSize: 11)),
          Text('CLAP ${clip.clapScore!.toStringAsFixed(3)}',
              style: const TextStyle(
                  color: NanoColors.pink,
                  fontSize: 11,
                  fontWeight: FontWeight.bold)),
        ],
        const Spacer(),
        if (ready) _miniIcon(Icons.download, 'download', onDownload),
        _miniIcon(Icons.close, 'remove', onDelete),
      ],
    );
  }

  Widget _miniIcon(IconData icon, String tip, VoidCallback onTap) {
    return IconButton(
      onPressed: onTap,
      icon: Icon(icon, size: 16),
      color: NanoColors.textDim,
      tooltip: tip,
      visualDensity: VisualDensity.compact,
      constraints: const BoxConstraints(),
      padding: const EdgeInsets.all(6),
    );
  }

  /// Visible, immediate-drag handle: click-drag it onto the create panel to set
  /// this clip as the extend source. A plain [Draggable] (not long-press) so a
  /// normal mouse drag starts it; the small handle target keeps it from fighting
  /// the list's vertical scroll.
  Widget _dragHandle() {
    return Draggable<GenClip>(
      data: clip,
      dragAnchorStrategy: pointerDragAnchorStrategy,
      feedback: _dragFeedback(),
      child: MouseRegion(
        cursor: SystemMouseCursors.grab,
        child: Tooltip(
          message: 'drag onto the create panel to extend, cover or split stems',
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: const [
              Icon(Icons.drag_indicator, size: 15, color: NanoColors.pink),
              SizedBox(width: 2),
              Text(
                'use',
                style: TextStyle(
                  color: NanoColors.pink,
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _dragFeedback() {
    return Material(
      color: Colors.transparent,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: NanoColors.pink,
          borderRadius: BorderRadius.circular(6),
        ),
        child: Text(
          clip.prompt.isEmpty ? 'nano-clip-${clip.id}' : clip.prompt,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(
            color: Colors.black,
            fontWeight: FontWeight.bold,
            fontSize: 12,
          ),
        ),
      ),
    );
  }
}
