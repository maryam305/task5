import os
import time
from playwright.sync_api import sync_playwright

def capture_all():
    os.makedirs("screenshots", exist_ok=True)
    
    with sync_playwright() as p:
        # Launch Chromium with a mobile viewport (e.g., iPhone 12 Pro dimensions)
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 390, "height": 844},
            device_scale_factor=3, # High resolution scale factor
            is_mobile=True,
            has_touch=True
        )
        page = context.new_page()
        
        print("Navigating to Morphy app...")
        page.goto("http://localhost:8000")
        
        # Wait for the Flutter app to load and initialize
        print("Waiting for app initialization...")
        time.sleep(6)
        
        # Capture Login Screen
        print("Capturing Login Screen...")
        page.screenshot(path="screenshots/01_login_screen.png")
        
        # Login interaction using Tab navigation
        print("Performing login...")
        page.keyboard.press("Tab")
        time.sleep(0.5)
        page.keyboard.type("antigravity")
        time.sleep(0.5)
        page.keyboard.press("Tab")
        time.sleep(0.5)
        page.keyboard.type("antigravity@gmail.com")
        time.sleep(0.5)
        page.keyboard.press("Tab")
        time.sleep(0.5)
        page.keyboard.press("Enter")
        
        # Wait for transition to Home Screen
        print("Waiting for Home Screen...")
        time.sleep(4)
        
        # Capture Home Screen - Joyful Theme
        print("Capturing Home Screen (Joyful Theme)...")
        page.screenshot(path="screenshots/02_home_joyful.png")
        
        # Go to Settings to change theme to Galaxy
        # Settings button is at top right: around x=350, y=35
        print("Navigating to Settings...")
        page.mouse.click(350, 35)
        time.sleep(2)
        
        # Capture Settings Screen
        print("Capturing Settings Screen...")
        page.screenshot(path="screenshots/03_settings_screen.png")
        
        # Click GALAXY theme chip
        # Viewport is 390x844. Theme chips are in a Wrap. 
        # GALAXY is usually the second chip. Let's click it.
        # Let's try coordinates for Galaxy chip: x=140, y=280
        print("Selecting Galaxy Theme...")
        page.mouse.click(140, 280)
        time.sleep(2)
        
        # Click back button (top left: around x=30, y=35)
        print("Going back to Home...")
        page.mouse.click(30, 35)
        time.sleep(2)
        
        # Capture Home Screen - Galaxy Theme
        print("Capturing Home Screen (Galaxy Theme)...")
        page.screenshot(path="screenshots/04_home_galaxy.png")
        
        # Click "Theme Shop" card (bottom left: around x=100, y=740)
        print("Navigating to Theme Shop...")
        page.mouse.click(100, 740)
        time.sleep(3)
        
        # Capture Shop Screen - Galaxy Theme
        print("Capturing Shop Screen (Galaxy Theme)...")
        page.screenshot(path="screenshots/05_shop_galaxy.png")
        
        # Scroll down in Shop to see the item list
        print("Scrolling Shop Screen...")
        page.mouse.wheel(0, 400)
        time.sleep(2)
        page.screenshot(path="screenshots/06_shop_items.png")
        
        # Go back to Home
        print("Going back to Home...")
        page.mouse.click(30, 35)
        time.sleep(2)
        
        # Go to Settings and select CYBERPUNK theme (third chip, around x=240, y=280)
        print("Navigating to Settings for Cyberpunk...")
        page.mouse.click(350, 35)
        time.sleep(2)
        print("Selecting Cyberpunk Theme...")
        page.mouse.click(240, 280)
        time.sleep(2)
        page.screenshot(path="screenshots/07_settings_cyberpunk.png")
        
        # Go back to Home
        print("Going back to Home...")
        page.mouse.click(30, 35)
        time.sleep(2)
        page.screenshot(path="screenshots/08_home_cyberpunk.png")
        
        # Click "Support" card (bottom right: around x=290, y=740)
        print("Navigating to Support...")
        page.mouse.click(290, 740)
        time.sleep(3)
        
        # Capture Support Screen
        print("Capturing Support Screen...")
        page.screenshot(path="screenshots/09_support_screen.png")
        
        print("All screenshots successfully captured!")
        browser.close()

if __name__ == "__main__":
    capture_all()
