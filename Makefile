.PHONY: start start-de test help

# Default app (English)
start:
	@echo "🚀 Starting RAG Application (English)..."
	@uv run streamlit run apps/app_v2_0.py --server.port 8501 --server.address localhost

# German version
start-de:
	@echo "🚀 Starting RAG Application (German)..."
	@uv run streamlit run apps/app_v2_0g.py --server.port 8501 --server.address localhost

# V1.1 version
start-v1:
	@echo "🚀 Starting RAG Application (V1.1)..."
	@uv run streamlit run apps/app_v1_1.py --server.port 8501 --server.address localhost

# Run tests
test:
	@echo "🧪 Running tests..."
	@uv run test_source_linking.py
	@uv run test_empty_response.py

# Run diagnostic
diagnose:
	@echo "🔍 Running source linking diagnostic..."
	@uv run diagnose_source_linking.py

# Test source linking in browser
test-sources:
	@echo "🔗 Starting source linking test app..."
	@uv run streamlit run dev/basic_report-source-tester_app.py --server.port 8502 --server.address localhost

# Show help
help:
	@echo "Available commands:"
	@echo "  make start        - Start the English version (default)"
	@echo "  make start-de     - Start the German version"
	@echo "  make start-v1     - Start the V1.1 version"
	@echo "  make test         - Run automated tests"
	@echo "  make diagnose     - Run source linking diagnostic"
	@echo "  make test-sources - Test source linking in browser (port 8502)"
	@echo "  make help         - Show this help message"
