import 'package:flutter_test/flutter_test.dart';
import 'package:nano_ui/api.dart';

void main() {
  group('NanoApi.streamUri', () {
    final api = NanoApi('https://diskrot--nano-serve-serve.modal.run');

    test('builds the /generate_stream GET URL with encoded params', () {
      final p = GenParams()
        ..prompt = 'punchy techno, groovy synth bass'
        ..lyrics = ''
        ..seconds = 30.0
        ..cfgScale = 7.0
        ..sweeten = true;
      final uri = api.streamUri(p, 'g1t123');

      expect(uri.path, '/generate_stream');
      expect(uri.host, 'diskrot--nano-serve-serve.modal.run');
      expect(uri.queryParameters['req_id'], 'g1t123');
      expect(uri.queryParameters['seconds'], '30.0');
      expect(uri.queryParameters['cfg_scale'], '7.0');
      expect(uri.queryParameters['sweeten'], 'true');
      // Spaces/commas in the prompt are URL-encoded, not raw, in the query string.
      expect(uri.query.contains('punchy techno'), isFalse);
      expect(uri.queryParameters['prompt'], 'punchy techno, groovy synth bass');
    });

    test('per-codebook ladders and gender ride through verbatim', () {
      final p = GenParams()
        ..perCbTemperature = '1.05,0.98,0.9'
        ..perCbTopK = '120,90,70'
        ..gender = 'female';
      final uri = api.streamUri(p, 'abc');
      expect(uri.queryParameters['per_cb_temperature'], '1.05,0.98,0.9');
      expect(uri.queryParameters['per_cb_top_k'], '120,90,70');
      expect(uri.queryParameters['gender'], 'female');
    });
  });
}
