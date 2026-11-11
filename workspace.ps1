# Opens and arranges the 3-monitor workspace.
#   Left monitor:  htop (left half) | claude --resume (right half)
#   Main monitor:  Zed (left two-thirds) | two terminals stacked (right third), bottom one attached to tmux "views"
#   Right monitor: terminal (top half) / Spotify (bottom half), Chrome on top of Spotify

$ErrorActionPreference = 'Stop'
$Dir = $PSScriptRoot
$Alacritty = 'C:\Program Files\Alacritty\alacritty.exe'
$Spotify = "$env:APPDATA\Spotify\Spotify.exe"
$Zed = "$env:LOCALAPPDATA\Programs\Zed\bin\Zed.exe"
$Chrome = 'C:\Program Files\Google\Chrome\Application\chrome.exe'

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class Win {
    [DllImport("user32.dll")] public static extern bool SetProcessDpiAwarenessContext(IntPtr v);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
    [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int w, int ht, uint flags);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("dwmapi.dll")] public static extern int DwmGetWindowAttribute(IntPtr h, int attr, out RECT r, int size);
    [DllImport("user32.dll")] static extern bool EnumWindows(EnumProc cb, IntPtr p);
    [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll")] static extern int GetWindowTextLength(IntPtr h);
    [DllImport("user32.dll")] static extern IntPtr GetWindow(IntPtr h, uint cmd);
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    delegate bool EnumProc(IntPtr h, IntPtr p);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }

    // Visible, titled, unowned top-level windows belonging to the given process IDs.
    public static IntPtr[] WindowsOf(int[] pids) {
        var set = new System.Collections.Generic.HashSet<int>(pids);
        var list = new System.Collections.Generic.List<IntPtr>();
        EnumWindows((h, p) => {
            uint pid; GetWindowThreadProcessId(h, out pid);
            if (set.Contains((int)pid) && IsWindowVisible(h) && GetWindowTextLength(h) > 0 && GetWindow(h, 4) == IntPtr.Zero)
                list.Add(h);
            return true;
        }, IntPtr.Zero);
        return list.ToArray();
    }
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
    # Alacritty briefly shows an untitled placeholder window at startup; wait for the real (titled) one
    Wait-Window { Get-Process -Id $termId -ErrorAction SilentlyContinue | Where-Object MainWindowTitle }.GetNewClosure()
}

# Kill whatever listens on a port, plus its whole process tree (so a --reload watcher can't respawn it)
function Stop-Port([int]$port) {
    $pids = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($id in $pids) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$id"
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.ParentProcessId)"
        if ($parent.Name -eq 'python.exe') { $id = $parent.ProcessId }   # walk up to the reloader
        taskkill /T /F /PID $id | Out-Null
    }
}

# tmux session "views": create it if missing (signal / oven / plain window).
# Commands are passed as separate words: psmux's new-window silently drops a single quoted command string.
tmux has-session -t views 2>$null
if ($LASTEXITCODE -ne 0) {
    Stop-Port 8052
    Stop-Port 8053
    tmux new-session -d -s views -n signal -c $Dir pwsh -NoLogo -NoExit -Command "python -m dashboard.app --host raptor --reload"
    tmux new-window -t views: -n oven -c $Dir pwsh -NoLogo -NoExit -Command "python -m oven --host raptor --port 8053"
    tmux new-window -t views: -c $Dir
}

# Launch everything
$htop    = Start-Term 'htop'
$claude  = Start-Term 'claude --resume'
$mainTop = Start-Term
$mainBot = Start-Term 'tmux attach -t views'
$rightTop = Start-Term

if (-not (Get-Process Zed -ErrorAction SilentlyContinue | Where-Object MainWindowHandle -ne 0)) {
    Start-Process $Zed -ArgumentList $Dir -WindowStyle Hidden
}
$zedWin = Wait-Window { Get-Process Zed -ErrorAction SilentlyContinue | Where-Object MainWindowTitle } 60

if (-not (Get-Process Spotify -ErrorAction SilentlyContinue | Where-Object MainWindowHandle -ne 0)) {
    Start-Process $Spotify
}
$spot = Wait-Window { Get-Process Spotify -ErrorAction SilentlyContinue | Where-Object MainWindowTitle } 60

# Chrome: open a new window and pick it out from any Chrome windows that already existed
function Get-ChromeWindows { [Win]::WindowsOf([int[]]@(Get-Process chrome -ErrorAction SilentlyContinue).Id) }
$chromeBefore = @(Get-ChromeWindows)
Start-Process $Chrome -ArgumentList '--new-window'
$deadline = (Get-Date).AddSeconds(30)
do {
    Start-Sleep -Milliseconds 200
    $chromeWin = Get-ChromeWindows | Where-Object { $_ -notin $chromeBefore } | Select-Object -First 1
} until ($chromeWin -or (Get-Date) -gt $deadline)
if (-not $chromeWin) { throw 'Chrome window did not appear in 30 s' }

# Left monitor: halves side by side
$half = [int]($L.Width / 2)
Place $htop   $L.X           $L.Y $half             $L.Height
Place $claude ($L.X + $half) $L.Y ($L.Width - $half) $L.Height

# Main monitor: Zed left two-thirds, two terminals stacked in the remaining third
$twoThirds = [int]($M.Width * 2 / 3)
$mHalf = [int]($M.Height / 2)
Place $zedWin  $M.X               $M.Y            $twoThirds              $M.Height
Place $mainTop ($M.X + $twoThirds) $M.Y            ($M.Width - $twoThirds) $mHalf
Place $mainBot ($M.X + $twoThirds) ($M.Y + $mHalf) ($M.Width - $twoThirds) ($M.Height - $mHalf)

# Right monitor: terminal top half, Spotify bottom half with Chrome on top of it
$rHalf = [int]($R.Height / 2)
Place $rightTop  $R.X $R.Y            $R.Width $rHalf
Place $spot      $R.X ($R.Y + $rHalf) $R.Width ($R.Height - $rHalf)
Place $chromeWin $R.X ($R.Y + $rHalf) $R.Width ($R.Height - $rHalf)
[Win]::SetWindowPos($chromeWin, [IntPtr]::Zero, 0, 0, 0, 0, 0x0013) | Out-Null   # raise above Spotify (NOSIZE|NOMOVE|NOACTIVATE)
