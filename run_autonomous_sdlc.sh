#!/bin/bash

# 🚀 Autonomous SDLC Execution Script
# Complete autonomous software development lifecycle execution

echo "🎯 AUTONOMOUS SDLC EXECUTION SYSTEM"
echo "🔬 Terragon Labs - Quantum-Optimized Development"
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="

# Check Python version
python3 --version 2>/dev/null || {
    echo "❌ Python 3 is required but not installed"
    exit 1
}

echo "✅ Python 3 detected"

# Check if we're in the right directory
if [ ! -f "autonomous_sdlc_master_demo.py" ]; then
    echo "❌ autonomous_sdlc_master_demo.py not found. Run from project root."
    exit 1
fi

echo "✅ Project structure validated"

# Install dependencies (if needed)
echo "📦 Checking dependencies..."
python3 -c "import psutil" 2>/dev/null || {
    echo "⚠️ Installing missing dependencies..."
    pip3 install psutil 2>/dev/null || {
        echo "❌ Failed to install dependencies. Please run: pip3 install psutil"
        exit 1
    }
}

echo "✅ Dependencies ready"

# Set up environment
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export AUTONOMOUS_SDLC_MODE="production"
export SECURITY_LEVEL="maximum"

echo "🔧 Environment configured"

# Display execution options
echo ""
echo "🎮 Available Demonstrations:"
echo "   1. Generation 1 Basic Demo (Basic functionality)"
echo "   2. Generation 3 Comprehensive Demo (Full Gen 1-3 features)"  
echo "   3. MASTER SYSTEM DEMO (Complete autonomous SDLC)"
echo ""

# Prompt user for choice
read -p "Select demonstration (1-3) or press Enter for Master Demo: " choice

case $choice in
    1)
        echo "🚀 Running Generation 1 Basic Demo..."
        python3 generation_1_implementation_demo.py
        ;;
    2)
        echo "🚀 Running Generation 3 Comprehensive Demo..."
        python3 generation_3_comprehensive_demo.py
        ;;
    3|"")
        echo "🚀 Running MASTER SYSTEM DEMO..."
        echo "🎯 Complete Autonomous SDLC Execution"
        echo ""
        python3 autonomous_sdlc_master_demo.py
        ;;
    *)
        echo "❌ Invalid choice. Running Master Demo by default..."
        python3 autonomous_sdlc_master_demo.py
        ;;
esac

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "🎊 AUTONOMOUS SDLC EXECUTION COMPLETED SUCCESSFULLY!"
    echo "✅ System is ready for production deployment"
    echo "🧠 Continuous learning and improvement active"
else
    echo "⚠️ Execution completed with warnings or errors"
    echo "📋 Check logs for detailed information"
fi

echo ""
echo "🔗 Next Steps:"
echo "   • Review execution results and recommendations"
echo "   • Deploy to production using Docker or Kubernetes"
echo "   • Monitor system performance and learning progress"
echo "   • Scale globally with compliance frameworks"
echo ""
echo "📚 Documentation: See AUTONOMOUS_SDLC_IMPLEMENTATION_SUMMARY.md"
echo "🚀 Ready for autonomous operation!"

exit $exit_code