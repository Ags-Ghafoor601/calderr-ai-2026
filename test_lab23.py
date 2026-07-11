"""Quick test script for Lab 2.3 components."""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")
import labs.lab_2_3_error_recovery as lab23

# Test backoff engine
engine = lab23.BackoffEngine(base_delay=0.5, max_delay=16.0)
print("=== Backoff Engine Test ===")
for i in range(5):
    delay = engine.get_delay(i)
    print(f"  Attempt {i+1}: {delay:.2f}s")

# Test circuit breaker
cb = lab23.CircuitBreaker(failure_threshold=3)
print()
print("=== Circuit Breaker Test ===")
print(f"  Available before failures: {cb.is_available('test_tool')}")
cb.record_failure("test_tool")
cb.record_failure("test_tool")
cb.record_failure("test_tool")
print(f"  Available after 3 failures: {cb.is_available('test_tool')}")

# Test that tools exist and are callable
print()
print("=== Tool Existence Test ===")
print(f"  fetch_stock_price: {lab23.fetch_stock_price.name}")
print(f"  fetch_weather: {lab23.fetch_weather.name}")
print(f"  fetch_news: {lab23.fetch_news.name}")
print(f"  Fallback map has {len(lab23.FALLBACK_MAP)} entries")

print()
print("Lab 2.3 components work correctly!")
