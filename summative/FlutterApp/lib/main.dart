import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

/// Public URL of the FastAPI service on Render (replace with your own service URL).
const String apiBaseUrl = '';

void main() => runApp(const FoodPriceApp());

class FoodPriceApp extends StatelessWidget {
  const FoodPriceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Kenya Food Price Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: Colors.green, useMaterial3: true),
      home: const PredictionPage(),
    );
  }
}

enum FieldKind { text, integer, decimal }

/// One input variable of the model: the JSON key the API expects, a label, a hint and its type.
class InputSpec {
  const InputSpec(this.key, this.label, this.hint, this.kind);
  final String key;
  final String label;
  final String hint;
  final FieldKind kind;
}

const List<InputSpec> inputSpecs = [
  InputSpec('commodity', 'Commodity', 'e.g. Beans (dry), Maize (white), Sorghum', FieldKind.text),
  InputSpec('pricetype', 'Price type', 'Retail or Wholesale', FieldKind.text),
  InputSpec('admin1', 'Region', 'e.g. North Eastern, Rift Valley, Nairobi', FieldKind.text),
  InputSpec('latitude', 'Market latitude', '-4.8 to 5.0, e.g. 0.11', FieldKind.decimal),
  InputSpec('longitude', 'Market longitude', '33.9 to 42.0, e.g. 40.32', FieldKind.decimal),
  InputSpec('year', 'Year', '2006 to 2030', FieldKind.integer),
  InputSpec('month', 'Month', '1 to 12', FieldKind.integer),
  InputSpec('pack_size', 'Quantity (kg or L)', 'e.g. 1, 50, 90', FieldKind.decimal),
];

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final Map<String, TextEditingController> _controllers = {
    for (final spec in inputSpecs) spec.key: TextEditingController(),
  };

  bool _loading = false;
  String? _result; // shown when the prediction succeeds
  String? _error; // shown in the same area when something is wrong

  @override
  void dispose() {
    for (final c in _controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  /// Checks for missing values and wrong types before calling the API.
  /// Range checks and valid category names are enforced by the API and shown from its response.
  Map<String, dynamic>? _buildBody(List<String> problems) {
    final body = <String, dynamic>{};
    for (final spec in inputSpecs) {
      final raw = _controllers[spec.key]!.text.trim();
      if (raw.isEmpty) {
        problems.add('${spec.label} is required.');
        continue;
      }
      switch (spec.kind) {
        case FieldKind.text:
          body[spec.key] = raw;
        case FieldKind.integer:
          final value = int.tryParse(raw);
          if (value == null) {
            problems.add('${spec.label} must be a whole number (got "$raw").');
          } else {
            body[spec.key] = value;
          }
        case FieldKind.decimal:
          final value = double.tryParse(raw);
          if (value == null) {
            problems.add('${spec.label} must be a number (got "$raw").');
          } else {
            body[spec.key] = value;
          }
      }
    }
    return problems.isEmpty ? body : null;
  }

  /// ---- The API call: POST the inputs as JSON to /predict ----
  Future<void> _predict() async {
    FocusScope.of(context).unfocus();
    final problems = <String>[];
    final body = _buildBody(problems);
    if (body == null) {
      setState(() {
        _result = null;
        _error = problems.join('\n');
      });
      return;
    }

    setState(() {
      _loading = true;
      _result = null;
      _error = null;
    });

    try {
      final response = await http
          .post(
            Uri.parse('$apiBaseUrl/predict'),
            headers: {'Content-Type': 'application/json', 'Accept': 'application/json'},
            body: jsonEncode(body),
          )
          .timeout(const Duration(seconds: 90)); // a sleeping free Render service can take ~1 minute to wake

      final data = jsonDecode(response.body);
      if (response.statusCode == 200) {
        final perKg = (data['predicted_price_per_kg'] as num).toDouble();
        final total = (data['predicted_total_price'] as num).toDouble();
        setState(() => _result =
            'KES ${perKg.toStringAsFixed(2)} per kg\n'
            'Total for ${body['pack_size']} kg: KES ${total.toStringAsFixed(2)}');
      } else if (response.statusCode == 422) {
        setState(() => _error = _readValidationErrors(data));
      } else {
        setState(() => _error = 'Server error (${response.statusCode}). Please try again.');
      }
    } on TimeoutException {
      setState(() => _error = 'The server took too long to respond. It may be waking up — try again.');
    } catch (e) {
      setState(() => _error = 'Could not reach the server. Check your internet connection.\n($e)');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  /// Turns FastAPI/Pydantic 422 details into readable lines, e.g. "Month: Input should be less than or equal to 12".
  String _readValidationErrors(dynamic data) {
    final detail = data is Map ? data['detail'] : null;
    if (detail is! List) return 'Invalid input.';
    return detail.map((d) {
      final loc = (d['loc'] as List).last.toString();
      final label = inputSpecs
          .firstWhere((s) => s.key == loc, orElse: () => InputSpec(loc, loc, '', FieldKind.text))
          .label;
      final msg = d['msg'].toString().replaceFirst('Value error, ', '');
      return '$label: $msg';
    }).join('\n');
  }

  Widget _field(InputSpec spec) {
    return TextField(
      controller: _controllers[spec.key],
      keyboardType: switch (spec.kind) {
        FieldKind.text => TextInputType.text,
        FieldKind.integer => TextInputType.number,
        FieldKind.decimal => const TextInputType.numberWithOptions(decimal: true, signed: true),
      },
      textInputAction: TextInputAction.next,
      decoration: InputDecoration(
        labelText: spec.label,
        helperText: spec.hint,
        border: const OutlineInputBorder(),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final hasError = _error != null;

    return Scaffold(
      appBar: AppBar(title: const Text('Kenya Food Price Predictor')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Estimate the market price of a crop before you sell.',
                style: theme.textTheme.bodyLarge,
              ),
              const SizedBox(height: 16),
              for (final spec in inputSpecs) ...[
                _field(spec),
                const SizedBox(height: 12),
              ],
              const SizedBox(height: 4),
              SizedBox(
                height: 48,
                child: FilledButton(
                  onPressed: _loading ? null : _predict,
                  child: _loading
                      ? const SizedBox(
                          width: 22, height: 22, child: CircularProgressIndicator(strokeWidth: 2.5))
                      : const Text('Predict', style: TextStyle(fontSize: 16)),
                ),
              ),
              const SizedBox(height: 20),
              // Display area: prediction or error message
              Container(
                padding: const EdgeInsets.all(16),
                constraints: const BoxConstraints(minHeight: 80),
                decoration: BoxDecoration(
                  color: hasError ? theme.colorScheme.errorContainer : theme.colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      hasError ? 'Please fix the input' : 'Predicted price',
                      style: theme.textTheme.titleSmall,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _error ?? _result ?? 'Fill in all fields and tap Predict.',
                      style: hasError
                          ? theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onErrorContainer)
                          : theme.textTheme.titleLarge,
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
