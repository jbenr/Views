# Opens Alacritty windows for the Views project and puts each one at a fixed spot on screen.
# Run with:  pwsh -File workspace.ps1   (or right-click > Run with PowerShell)
#
# Your monitors (pixels):
#   Left    1920x1080  X=-1920..0     Y=212..1292
#   Middle  3440x1440  X=0..3440      Y=0..1440     (primary)
#   Right   1080x1920  X=3440..4520   Y=-487..1433  (vertical)

$Project = "C:\Users\benjils\werk\Views"

# One entry per window. X/Y = top-left corner, W/H = size, Cmd = what to run (blank = plain shell).
$Windows = @(
    @{ Title = "claude"; X = 0;    Y = 0;    W = 1720; H = 1440; Cmd = "claude" }
    @{ Title = "shell";  X = 1720; Y = 0;    W = 1720; H = 720;  Cmd = "conda activate 2s10s" }
    @{ Title = "git";    X = 1720; Y = 720;  W = 1720; H = 720;  Cmd = "git status" }
    @{ Title = "logs";   X = 3440; Y = -487; W = 1080; H = 1920; Cmd = "" }
)

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class Win {
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr h, int x, int y, int w, int ht, bool repaint);
}
"@
[Win]::SetProcessDPIAware() | Out-Null   # work in real pixels on every monitor

foreach ($w in $Windows) {
    $argLine = "--title `"$($w.Title)`" -o window.dynamic_title=false --working-directory `"$Project`" -e pwsh -NoExit"
    if ($w.Cmd) { $argLine += " -Command `"$($w.Cmd)`"" }

    $proc = Start-Process alacritty -ArgumentList $argLine -PassThru

    # Wait (up to 5s) for the real window (Alacritty briefly shows a temporary one first), then move it.
    for ($i = 0; $i -lt 50 -and $proc.MainWindowTitle -ne $w.Title; $i++) {
        Start-Sleep -Milliseconds 100
        $proc.Refresh()
    }
    if ($proc.MainWindowTitle -eq $w.Title) {
        [Win]::MoveWindow($proc.MainWindowHandle, $w.X, $w.Y, $w.W, $w.H, $true) | Out-Null
    } else {
        Write-Warning "Window '$($w.Title)' didn't appear in time; left it where Alacritty put it."
    }
}
