import 'package:flutter/material.dart';

/// nano palette — white on black with hot-pink highlights.
class NanoColors {
  static const Color bg = Color(0xFF000000);
  static const Color surface = Color(0xFF0A0A0A);
  static const Color surfaceAlt = Color(0xFF121212);
  static const Color border = Color(0xFF242424);
  static const Color text = Color(0xFFF5F5F5);
  static const Color textDim = Color(0xFF8A8A8A);
  static const Color pink = Color(0xFFFF2D95);
  static const Color pinkDim = Color(0xFF7A1648);
  static const Color error = Color(0xFFFF4D4D);
  static const Color ok = Color(0xFF35E08A);
}

ThemeData buildNanoTheme() {
  const pink = NanoColors.pink;
  final base = ThemeData.dark(useMaterial3: true);

  return base.copyWith(
    scaffoldBackgroundColor: NanoColors.bg,
    canvasColor: NanoColors.bg,
    colorScheme: const ColorScheme.dark(
      primary: pink,
      secondary: pink,
      surface: NanoColors.surface,
      onSurface: NanoColors.text,
      error: NanoColors.error,
    ),
    textTheme: base.textTheme.apply(
      bodyColor: NanoColors.text,
      displayColor: NanoColors.text,
      fontFamily: 'monospace',
    ),
    sliderTheme: base.sliderTheme.copyWith(
      activeTrackColor: pink,
      inactiveTrackColor: NanoColors.border,
      thumbColor: pink,
      overlayColor: pink.withValues(alpha: 0.15),
      trackHeight: 2,
      valueIndicatorColor: pink,
      valueIndicatorTextStyle: const TextStyle(
        color: Colors.black,
        fontFamily: 'monospace',
        fontWeight: FontWeight.bold,
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: NanoColors.surfaceAlt,
      hintStyle: const TextStyle(color: NanoColors.textDim),
      contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(4),
        borderSide: const BorderSide(color: NanoColors.border),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(4),
        borderSide: const BorderSide(color: pink, width: 1.5),
      ),
    ),
    switchTheme: SwitchThemeData(
      thumbColor: WidgetStateProperty.resolveWith(
        (s) => s.contains(WidgetState.selected) ? pink : NanoColors.textDim,
      ),
      trackColor: WidgetStateProperty.resolveWith(
        (s) => s.contains(WidgetState.selected)
            ? NanoColors.pinkDim
            : NanoColors.border,
      ),
      trackOutlineColor: WidgetStateProperty.all(Colors.transparent),
    ),
  );
}
