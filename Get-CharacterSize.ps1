Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Screen {
    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();
    
    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, ref RECT rect);
    
    public struct RECT {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }

    public static RECT GetWindowRect() {
        IntPtr hWnd = GetForegroundWindow();
        RECT rect = new RECT();
        GetWindowRect(hWnd, ref rect);
        return rect;
    }
}
"@

$rect = [Screen]::GetWindowRect()
$windowWidth = $rect.Right - $rect.Left
$windowHeight = $rect.Bottom - $rect.Top
Write-Output "Window Width: $windowWidth pixels"
Write-Output "Window Height: $windowHeight pixels"

$buffer = $host.UI.RawUI.BufferSize
$charWidth = $windowWidth / $buffer.Width
$charHeight = $windowHeight / $buffer.Height
Write-Output "Character Width: $charWidth pixels"
Write-Output "Character Height: $charHeight pixels"
write-output "Character Aspect Ratio: $($charWidth / $charHeight)"