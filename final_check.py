import os
import subprocess
import sys

class Colors:
    HEADER = '\033[95m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def run_check(name, cmd):
    print(f"{Colors.OKCYAN}Running {name}...{Colors.ENDC}", end=" ")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"{Colors.OKGREEN}PASSED{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.FAIL}FAILED{Colors.ENDC}")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"{Colors.FAIL}ERROR ({e}){Colors.ENDC}")
        return False

def check_file(filename):
    print(f"{Colors.OKCYAN}Checking for {filename}...{Colors.ENDC}", end=" ")
    if os.path.exists(filename):
        print(f"{Colors.OKGREEN}FOUND{Colors.ENDC}")
        return True
    else:
        print(f"{Colors.FAIL}MISSING{Colors.ENDC}")
        return False

def main():
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
        
    print(f"\n{Colors.HEADER}{Colors.BOLD}=============================={Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}🚀 FINAL SUBMISSION CHECK{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}=============================={Colors.ENDC}\n")

    passed = True

    # Check Artifacts
    if not check_file("reward_curve.png"): passed = False
    if not check_file("trust_calibration_curve.png"): passed = False
    if not check_file("README.md"): passed = False

    # Check Demo
    if not run_check("demo.py", f'"{sys.executable}" demo.py'): passed = False

    # Check Validation
    if not run_check("validate_env.py", f'"{sys.executable}" validate_env.py'): passed = False

    # Readme Link Check
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
            if "youtube.com" not in content.lower() and "youtu.be" not in content.lower() and "loom.com" not in content.lower():
                print(f"{Colors.WARNING}WARNING: No video link found in README.md{Colors.ENDC}")
            if "[Link to Demo Video]" in content:
                print(f"{Colors.WARNING}WARNING: Placeholder '[Link to Demo Video]' still in README.md{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}Could not read README.md: {e}{Colors.ENDC}")
        passed = False

    print(f"\n{Colors.HEADER}{Colors.BOLD}=============================={Colors.ENDC}")
    if passed:
        print(f"{Colors.OKGREEN}{Colors.BOLD}✅ READY FOR SUBMISSION{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}❌ CHECKS FAILED - DO NOT SUBMIT YET{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}=============================={Colors.ENDC}\n")

if __name__ == "__main__":
    main()
