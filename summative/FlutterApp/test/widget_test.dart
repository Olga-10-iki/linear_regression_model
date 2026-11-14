import 'package:flutter_test/flutter_test.dart';

import 'package:food_price_app/main.dart';

void main() {
  testWidgets('shows all inputs, the Predict button and a missing-value error', (tester) async {
    await tester.pumpWidget(const FoodPriceApp());

    for (final spec in inputSpecs) {
      expect(find.text(spec.label), findsOneWidget);
    }
    expect(find.text('Predict'), findsOneWidget);

    await tester.ensureVisible(find.text('Predict'));
    await tester.tap(find.text('Predict'));
    await tester.pump();
    expect(find.textContaining('Commodity is required.'), findsOneWidget);
  });
}
