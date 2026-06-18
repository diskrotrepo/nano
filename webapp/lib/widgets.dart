import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'audio_web.dart';
import 'theme.dart';

/// A titled panel grouping related controls.
class SectionCard extends StatefulWidget {
  const SectionCard({
    super.key,
    required this.title,
    required this.children,
    this.collapsible = false,
    this.initiallyExpanded = true,
  });
  final String title;
  final List<Widget> children;

  /// When true the card header toggles the body open/closed.
  final bool collapsible;

  /// Starting state when [collapsible]; ignored otherwise.
  final bool initiallyExpanded;

  @override
  State<SectionCard> createState() => _SectionCardState();
}

class _SectionCardState extends State<SectionCard> {
  late bool _expanded = widget.initiallyExpanded;

  @override
  Widget build(BuildContext context) {
    final open = !widget.collapsible || _expanded;
    final titleText = Text(
      widget.title.toUpperCase(),
      style: const TextStyle(
        color: NanoColors.pink,
        fontSize: 11,
        letterSpacing: 2,
        fontWeight: FontWeight.bold,
      ),
    );
    final header = widget.collapsible
        ? InkWell(
            onTap: () => setState(() => _expanded = !_expanded),
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 2),
              child: Row(
                children: [
                  Expanded(child: titleText),
                  Icon(
                    open ? Icons.expand_less : Icons.expand_more,
                    size: 18,
                    color: NanoColors.pink,
                  ),
                ],
              ),
            ),
          )
        : titleText;

    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.fromLTRB(18, 16, 18, 18),
      decoration: BoxDecoration(
        color: NanoColors.surface,
        border: Border.all(color: NanoColors.border),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          header,
          if (open) ...[
            const SizedBox(height: 14),
            ...widget.children,
          ],
        ],
      ),
    );
  }
}

/// A slider with a label on the left and the live value on the right.
class LabeledSlider extends StatelessWidget {
  const LabeledSlider({
    super.key,
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.onChanged,
    this.divisions,
    this.fractionDigits = 2,
    this.suffix = '',
    this.help,
    this.minLabel,
    this.maxLabel,
    this.describe,
    this.defaultValue,
  });

  final String label;
  final double value;
  final double min;
  final double max;
  final ValueChanged<double> onChanged;
  final int? divisions;
  final int fractionDigits;
  final String suffix;
  final String? help;

  /// Marker shown at the left (min) end of the track.
  final String? minLabel;

  /// Marker shown at the right (max) end of the track.
  final String? maxLabel;

  /// Maps the current value to a short musical description shown over the track.
  final String Function(double value)? describe;

  /// When set, shows a reset button that restores the slider to this value.
  final double? defaultValue;

  @override
  Widget build(BuildContext context) {
    final clamped = value.clamp(min, max);
    final frac = max > min ? ((clamped - min) / (max - min)).clamp(0.0, 1.0) : 0.0;
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(child: _FieldLabel(label, help: help)),
              Text(
                '${value.toStringAsFixed(fractionDigits)}$suffix',
                style: const TextStyle(
                  color: NanoColors.pink,
                  fontWeight: FontWeight.bold,
                ),
              ),
              if (defaultValue != null) ...[
                const SizedBox(width: 4),
                InkWell(
                  onTap: (value - defaultValue!).abs() < 1e-9
                      ? null
                      : () => onChanged(defaultValue!),
                  borderRadius: BorderRadius.circular(4),
                  child: Padding(
                    padding: const EdgeInsets.all(2),
                    child: Icon(
                      Icons.refresh,
                      size: 15,
                      color: (value - defaultValue!).abs() < 1e-9
                          ? NanoColors.border
                          : NanoColors.textDim,
                      semanticLabel: 'reset to default',
                    ),
                  ),
                ),
              ],
            ],
          ),
          if (describe != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 10),
              child: Align(
                alignment: Alignment(frac * 2 - 1, 0),
                child: Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: NanoColors.pinkDim,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(
                    describe!(clamped),
                    style: const TextStyle(
                      color: NanoColors.text,
                      fontSize: 11,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ),
            ),
          Slider(
            value: clamped,
            min: min,
            max: max,
            divisions: divisions,
            onChanged: onChanged,
          ),
          if (minLabel != null || maxLabel != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 10),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    minLabel ?? '',
                    style: const TextStyle(
                        color: NanoColors.textDim, fontSize: 10),
                  ),
                  Text(
                    maxLabel ?? '',
                    style: const TextStyle(
                        color: NanoColors.textDim, fontSize: 10),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}

class NanoTextField extends StatelessWidget {
  const NanoTextField({
    super.key,
    required this.label,
    required this.controller,
    this.hint,
    this.maxLines = 1,
    this.help,
    this.keyboardType,
    this.inputFormatters,
    this.onChanged,
  });

  final String label;
  final TextEditingController controller;
  final String? hint;
  final int maxLines;
  final String? help;
  final TextInputType? keyboardType;
  final List<TextInputFormatter>? inputFormatters;
  final ValueChanged<String>? onChanged;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _FieldLabel(label, help: help),
          const SizedBox(height: 6),
          TextField(
            controller: controller,
            maxLines: maxLines,
            keyboardType: keyboardType,
            inputFormatters: inputFormatters,
            onChanged: onChanged,
            style: const TextStyle(color: NanoColors.text, fontSize: 14),
            decoration: InputDecoration(hintText: hint),
          ),
        ],
      ),
    );
  }
}

class ToggleRow extends StatelessWidget {
  const ToggleRow({
    super.key,
    required this.label,
    required this.value,
    required this.onChanged,
    this.help,
  });

  final String label;
  final bool value;
  final ValueChanged<bool> onChanged;
  final String? help;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Expanded(child: _FieldLabel(label, help: help)),
          Switch(value: value, onChanged: onChanged),
        ],
      ),
    );
  }
}

/// Banner painted over the controls/create area while a clip is dragged onto
/// it. [IgnorePointer] so it never swallows the drop — the DragTarget beneath
/// must receive it.
class DropOverlay extends StatelessWidget {
  const DropOverlay({super.key, required this.label});
  final String label;

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: Container(
        decoration: BoxDecoration(
          color: NanoColors.pink.withValues(alpha: 0.12),
          border: Border.all(color: NanoColors.pink, width: 2),
          borderRadius: BorderRadius.circular(6),
        ),
        alignment: Alignment.center,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          decoration: BoxDecoration(
            color: NanoColors.pink,
            borderRadius: BorderRadius.circular(6),
          ),
          child: Text(
            label,
            style: const TextStyle(
              color: Colors.black,
              fontWeight: FontWeight.w900,
              letterSpacing: 1,
              fontSize: 14,
            ),
          ),
        ),
      ),
    );
  }
}

/// A library clip's mini-waveform scrubber. Decodes peaks from [bytes] once,
/// paints the bars with a playhead at the shared [player]'s progress when this
/// clip is current, and seeks on tap / horizontal drag.
class ClipWaveform extends StatefulWidget {
  const ClipWaveform({
    super.key,
    required this.id,
    required this.url,
    required this.bytes,
    required this.player,
    this.height = 40,
  });
  final String id;
  final String url;
  final Uint8List bytes;
  final ClipPlayer player;
  final double height;

  @override
  State<ClipWaveform> createState() => _ClipWaveformState();
}

class _ClipWaveformState extends State<ClipWaveform> {
  List<double>? _peaks;

  @override
  void initState() {
    super.initState();
    _decode();
  }

  Future<void> _decode() async {
    List<double> peaks;
    try {
      peaks = await decodeWaveformPeaks(widget.bytes, 160);
    } catch (_) {
      peaks = const [];
    }
    if (mounted) setState(() => _peaks = peaks);
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: widget.height,
      child: LayoutBuilder(
        builder: (context, constraints) {
          final w = constraints.maxWidth;
          void seekAt(double dx) =>
              widget.player.seek(widget.id, widget.url, dx / w);
          return GestureDetector(
            onTapDown: (d) => seekAt(d.localPosition.dx),
            onHorizontalDragStart: (d) => seekAt(d.localPosition.dx),
            onHorizontalDragUpdate: (d) => seekAt(d.localPosition.dx),
            child: AnimatedBuilder(
              animation: widget.player,
              builder: (context, _) {
                final progress = widget.player.isCurrent(widget.id)
                    ? widget.player.progress
                    : 0.0;
                return CustomPaint(
                  size: Size.infinite,
                  painter:
                      WaveformBarsPainter(peaks: _peaks, progress: progress),
                );
              },
            ),
          );
        },
      ),
    );
  }
}

/// The extend cut widget: the source clip's waveform with a draggable vertical
/// marker showing where the extension begins. Bars before the marker are kept;
/// after it they're greyed (discarded + regenerated). A timestamp label tracks
/// the marker. Reports the cut in seconds via [onChanged] — or `-1.0` (tail)
/// when the marker is dragged to the end, for a seamless lossless append.
class ExtendWaveform extends StatefulWidget {
  const ExtendWaveform({
    super.key,
    required this.bytes,
    required this.durationSeconds,
    required this.fromSeconds,
    required this.onChanged,
    this.height = 64,
  });
  final Uint8List bytes;
  final double durationSeconds;
  final double fromSeconds; // -1 = tail
  final ValueChanged<double> onChanged; // -1 = tail, else seconds
  final double height;

  @override
  State<ExtendWaveform> createState() => _ExtendWaveformState();
}

class _ExtendWaveformState extends State<ExtendWaveform> {
  List<double>? _peaks;
  double _decodedDuration = 0;

  /// True clip length: prefer the duration read from the decoded buffer (always
  /// correct) over the passed-in [durationSeconds] (may be 0 if never measured).
  double get _duration =>
      _decodedDuration > 0 ? _decodedDuration : widget.durationSeconds;

  @override
  void initState() {
    super.initState();
    _decode();
  }

  @override
  void didUpdateWidget(ExtendWaveform old) {
    super.didUpdateWidget(old);
    if (!identical(old.bytes, widget.bytes)) _decode();
  }

  Future<void> _decode() async {
    List<double> peaks;
    double duration = 0;
    try {
      final r = await decodeWaveform(widget.bytes, 240);
      peaks = r.peaks;
      duration = r.duration;
    } catch (_) {
      peaks = const [];
    }
    if (mounted) {
      setState(() {
        _peaks = peaks;
        _decodedDuration = duration;
      });
    }
  }

  double get _frac {
    final d = _duration;
    if (d <= 0 || widget.fromSeconds < 0) return 1.0;
    return (widget.fromSeconds / d).clamp(0.0, 1.0);
  }

  @override
  Widget build(BuildContext context) {
    final atTail = widget.fromSeconds < 0;
    final secs = atTail ? _duration : widget.fromSeconds;
    final label =
        '${atTail ? 'append' : 'cut'} @ ${_fmtClock(secs)} / ${_fmtClock(_duration)}';
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SizedBox(
          height: 16,
          child: Align(
            alignment: Alignment(_frac * 2 - 1, 0),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
              decoration: BoxDecoration(
                color: NanoColors.pink,
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                label,
                style: const TextStyle(
                  color: Colors.black,
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 4),
        SizedBox(
          height: widget.height,
          child: LayoutBuilder(
            builder: (context, constraints) {
              final w = constraints.maxWidth;
              void setAt(double dx) {
                final f = (dx / w).clamp(0.0, 1.0);
                // Snap to tail near the end → -1 (lossless seamless append).
                if (f >= 0.985 || _duration <= 0) {
                  widget.onChanged(-1.0);
                } else {
                  widget.onChanged(f * _duration);
                }
              }

              return GestureDetector(
                onTapDown: (d) => setAt(d.localPosition.dx),
                onHorizontalDragStart: (d) => setAt(d.localPosition.dx),
                onHorizontalDragUpdate: (d) => setAt(d.localPosition.dx),
                child: CustomPaint(
                  size: Size.infinite,
                  painter: WaveformBarsPainter(peaks: _peaks, cut: _frac),
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}

/// `m:ss` clock for waveform timestamps.
String _fmtClock(double s) {
  if (s.isNaN || s <= 0) return '0:00';
  final m = s ~/ 60;
  final sec = (s % 60).floor().toString().padLeft(2, '0');
  return '$m:$sec';
}

class _FieldLabel extends StatelessWidget {
  const _FieldLabel(this.text, {this.help});
  final String text;
  final String? help;

  @override
  Widget build(BuildContext context) {
    final label = Text(
      text,
      style: const TextStyle(color: NanoColors.text, fontSize: 13),
    );
    if (help == null) return label;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Flexible(child: label),
        const SizedBox(width: 6),
        Tooltip(
          message: help!,
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
    );
  }
}
