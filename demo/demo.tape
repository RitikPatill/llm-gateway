Output demo/demo.gif
Set FontSize 14
Set Width 1200
Set Height 600

# --- Start the gateway in the background ---
Type "uvicorn llm_gateway.main:app --reload &"
Enter
Sleep 3s

# --- First request: cache miss (full round-trip) ---
# Split across multiple Type calls to avoid \" escaping (unsupported in VHS 0.11.0 on Windows).
# Single-quoted VHS strings allow literal double-quotes; double-quoted VHS strings allow literal single-quotes.
Type "curl -s -X POST http://localhost:8000/v1/chat/completions -H "
Type "'"
Type "Content-Type: application/json"
Type "'"
Type " -d "
Type "'"
Type '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"ping"}],"stream":false}'
Type "'"
Enter
Sleep 2s

# --- Second request: cache hit (<5 ms) ---
Type "curl -s -X POST http://localhost:8000/v1/chat/completions -H "
Type "'"
Type "Content-Type: application/json"
Type "'"
Type " -d "
Type "'"
Type '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"ping"}],"stream":false}'
Type "'"
Enter
Sleep 1s

# --- Stop the background server, then open the Rich dashboard ---
Ctrl+C
Sleep 500ms
Type "python -m llm_gateway.dashboard"
Enter
Sleep 3s
