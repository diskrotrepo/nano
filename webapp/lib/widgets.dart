import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

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

/// A horizontal segmented selector (used for mode + seed mode).
class SegmentedRow<T> extends StatelessWidget {
  const SegmentedRow({
    super.key,
    required this.options,
    required this.selected,
    required this.onSelect,
    required this.labelOf,
  });

  final List<T> options;
  final T selected;
  final ValueChanged<T> onSelect;
  final String Function(T) labelOf;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: NanoColors.border),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Row(
        children: [
          for (final o in options)
            Expanded(
              child: GestureDetector(
                onTap: () => onSelect(o),
                child: Container(
                  padding: const EdgeInsets.symmetric(vertical: 11),
                  decoration: BoxDecoration(
                    color: o == selected ? NanoColors.pink : Colors.transparent,
                  ),
                  child: Center(
                    child: Text(
                      labelOf(o),
                      style: TextStyle(
                        color: o == selected ? Colors.black : NanoColors.textDim,
                        fontWeight: FontWeight.bold,
                        fontSize: 13,
                        letterSpacing: 1,
                      ),
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
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
