# run_day1.py

from app.data_loader import initialize_database

if __name__ == "__main__":
    print("🚀 ELEPHANT CONFLICT EARLY WARNING SYSTEM - DAY 1")
    print("==================================================")
    
    # This single function does everything:
    # 1. Drops old tables
    # 2. Creates new tables
    # 3. Fills them with synthetic data
    initialize_database()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Run: uvicorn main:app --reload")
    print("2. Run (in a new terminal): python run_day2.py")
    print("3. Run (in same new terminal): streamlit run dashboard.py")