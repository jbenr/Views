# Opens and arranges the 3-monitor workspace.
#   Left monitor:  htop (left half) | claude --resume (right half)
#   Main monitor:  VS Code (left two-thirds) | two terminals stacked (right third)
#   Right monitor: terminal (top half) / Spotify (bottom half)

$ErrorActionPreference = 'Stop'
$Dir = $PSScriptRoot
$Alacritty = 'C:\Program Files\Alacritty\alacritty.exe'
$Spotify = "$env:APPDATA\Spotify\Spotify.exe"

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class Win {
    [DllImport("user32.dll")] public static extern bool SetProcessDpiAwarenessContext(IntPtr v);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
    [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int w, int ht, uint flags);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("dwmapi.dll")] public static extern int DwmGetWindowAttribute(IntPtr h, int attr, out RECT r, int size);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
}
"@
[Win]::SetProcessDpiAwarenessContext([IntPtr]-4) | Out-Null   # per-monitor DPI aware -> physical pixels
Add-Type -AssemblyName System.Windows.Forms

# Monitors by position (work area excludes the taskbar)
$screens = [System.Windows.Forms.Screen]::AllScreens | Sort-Object { $_.Bounds.X }
$L = $screens[0].WorkingArea
$M = ($screens | Where-Object Primary).WorkingArea
$R = $screens[-1].WorkingArea

function Wait-Window([scriptblock]$find, [int]$timeoutSec = 30) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        $hit = & $find | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
        if ($hit) { return $hit.MainWindowHandle }
        Start-Sleep -Milliseconds 200
    }
    throw "Window did not appear in $timeoutSec s"
}

# Move a window so its *visible* frame fills (x, y, w, h), compensating for Win11's invisible borders.
function Place([IntPtr]$h, [int]$x, [int]$y, [int]$w, [int]$ht) {
    [Win]::ShowWindow($h, 9) | Out-Null   # SW_RESTORE (un-maximize)
    $flags = 0x0014                       # SWP_NOZORDER | SWP_NOACTIVATE
    [Win]::SetWindowPos($h, [IntPtr]::Zero, $x, $y, $w, $ht, $flags) | Out-Null
    $outer = New-Object Win+RECT; $vis = New-Object Win+RECT
    [Win]::GetWindowRect($h, [ref]$outer) | Out-Null
    [Win]::DwmGetWindowAttribute($h, 9, [ref]$vis, 16) | Out-Null   # DWMWA_EXTENDED_FRAME_BOUNDS
    $dl = $vis.L - $outer.L; $dt = $vis.T - $outer.T
    $dr = $outer.R - $vis.R; $db = $outer.B - $vis.B
    [Win]::SetWindowPos($h, [IntPtr]::Zero, $x - $dl, $y - $dt, $w + $dl + $dr, $ht + $dt + $db, $flags) | Out-Null
}

function Start-Term([string]$command) {
    $argList = @('--working-directory', $Dir)
    if ($command) { $argList += @('-e', 'pwsh.exe', '-NoLogo', '-NoExit', '-Command', $command) }
    $p = Start-Process $Alacritty -ArgumentList $argList -PassThru
    $termId = $p.Id
    Wait-Window { Get-Process -Id $termId -ErrorAction SilentlyContinue }.GetNewClosure()
}

# Launch everything
$htop    = Start-Term 'htop'
$claude  = Start-Term 'claude --resume'
$mainTop = Start-Term
$mainBot = Start-Term
$rightTop = Start-Term

if (-not (Get-Process Code -ErrorAction SilentlyContinue | Where-Object MainWindowHandle -ne 0)) {
    Start-Process code -ArgumentList $Dir -WindowStyle Hidden
}
$code = Wait-Window { Get-Process Code -ErrorAction SilentlyContinue | Where-Object MainWindowTitle } 60

if (-not (Get-Process Spotify -ErrorAction SilentlyContinue | Where-Object MainWindowHandle -ne 0)) {
    Start-Process $Spotify
}
$spot = Wait-Window { Get-Process Spotify -ErrorAction SilentlyContinue | Where-Object MainWindowTitle } 60

# Left monitor: halves side by side
$half = [int]($L.Width / 2)
Place $htop   $L.X           $L.Y $half             $L.Height
Place $claude ($L.X + $half) $L.Y ($L.Width - $half) $L.Height

# Main monitor: VS Code left two-thirds, two terminals stacked in the remaining third
$twoThirds = [int]($M.Width * 2 / 3)
$mHalf = [int]($M.Height / 2)
Place $code    $M.X                $M.Y            $twoThirds              $M.Height
Place $mainTop ($M.X + $twoThirds) $M.Y            ($M.Width - $twoThirds) $mHalf
Place $mainBot ($M.X + $twoThirds) ($M.Y + $mHalf) ($M.Width - $twoThirds) ($M.Height - $mHalf)

# Right monitor: terminal top half, Spotify bottom half
$rHalf = [int]($R.Height / 2)
Place $rightTop $R.X $R.Y            $R.Width $rHalf
Place $spot     $R.X ($R.Y + $rHalf) $R.Width ($R.Height - $rHalf)
