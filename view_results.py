#!/usr/bin/env python3
"""
Simple script to view TruLens evaluation results from persistent storage.
"""

import sys
import os
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
load_dotenv('.env.local')

from trulens.core import TruSession


def view_results():
    """View evaluation results from persistent storage."""
    session = TruSession()
    
    print("📊 TruLens Evaluation Results")
    print("=" * 50)
    
    # Get leaderboard
    leaderboard = session.get_leaderboard()
    print("\n🏆 Leaderboard:")
    print(leaderboard)
    
    # Show number of apps and records
    apps = session.get_apps()
    print(f"\n� Total Apps: {len(apps)}")
    
    if len(apps) > 0:
        print("\n🔍 Apps Summary:")
        for app in apps:
            app_name = app.get('app_name', 'Unknown')
            app_version = app.get('app_version', 'Unknown')
            print(f"  • {app_name} v{app_version}")


def launch_dashboard():
    """Launch the interactive TruLens dashboard."""
    session = TruSession()
    
    print("🚀 Launching TruLens interactive dashboard...")
    print("Press Ctrl+C to stop the dashboard")
    
    try:
        session.run_dashboard()
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="View TruLens evaluation results")
    parser.add_argument("--dashboard", action="store_true", help="Launch interactive dashboard")
    
    args = parser.parse_args()
    
    if args.dashboard:
        launch_dashboard()
    else:
        view_results()
