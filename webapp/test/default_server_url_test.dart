import 'package:flutter_test/flutter_test.dart';
import 'package:nano_ui/api.dart';

void main() {
  group('defaultServerUrl', () {
    test('modal dev hostname maps -ui to the sibling -serve URL', () {
      expect(
        defaultServerUrl('diskrot--nano-serve-ui-dev.modal.run'),
        'https://diskrot--nano-serve-serve-dev.modal.run',
      );
    });

    test('deployed modal hostname maps without the -dev suffix', () {
      expect(
        defaultServerUrl('diskrot--nano-serve-ui.modal.run'),
        'https://diskrot--nano-serve-serve.modal.run',
      );
    });

    test('localhost falls back to the local server', () {
      expect(defaultServerUrl('localhost'), 'http://127.0.0.1:8000');
      expect(defaultServerUrl('127.0.0.1'), 'http://127.0.0.1:8000');
    });

    test('non-ui modal hostnames fall back to the local server', () {
      expect(
        defaultServerUrl('diskrot--nano-serve-serve-dev.modal.run'),
        'http://127.0.0.1:8000',
      );
      expect(defaultServerUrl('example.com'), 'http://127.0.0.1:8000');
    });
  });
}
