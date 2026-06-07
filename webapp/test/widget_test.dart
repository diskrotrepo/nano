import 'package:flutter_test/flutter_test.dart';
import 'package:nano_ui/main.dart';

void main() {
  testWidgets('app renders the nano wordmark and mode selector',
      (WidgetTester tester) async {
    await tester.pumpWidget(const NanoApp());
    // Don't pump-and-settle: the post-frame /health check fires a network
    // request that never resolves under test.
    await tester.pump();

    expect(find.textContaining('diskrot'), findsOneWidget);
    expect(find.text('generate'), findsWidgets);
    expect(find.text('extend'), findsWidgets);
  });
}
