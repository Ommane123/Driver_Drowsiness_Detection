import sys
from ui import DrowsinessApp

def main():
    print("[SYSTEM] Starting ADAS Driver Safety Monitor...")
    app = DrowsinessApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

if __name__ == "__main__":
    main()
