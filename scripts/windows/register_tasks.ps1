# Run this script ONCE (as Administrator) to register Hogan's scheduled tasks.
# After registration these tasks run automatically — no terminal needed.
#
# Usage (in PowerShell as Admin):
#   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser   # one-time
#   cd C:\Users\15125\Documents\Hogan\Hogan\scripts
#   .\register_tasks.ps1

$HOGAN_ROOT = "C:\Users\15125\Documents\Hogan\Hogan"
$PS_EXE     = "C:\Program Files\PowerShell\7\pwsh.exe"  # PowerShell 7
if (-not (Test-Path $PS_EXE)) {
    $PS_EXE = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
}

# ── Task 1: Daily data refresh at 7:00 AM ────────────────────────────────────
$task1_name   = "Hogan_DailyRefresh"
$task1_script = "$HOGAN_ROOT\scripts\daily_refresh.ps1"
$task1_action = New-ScheduledTaskAction `
    -Execute $PS_EXE `
    -Argument "-NonInteractive -File `"$task1_script`"" `
    -WorkingDirectory $HOGAN_ROOT

$task1_trigger = New-ScheduledTaskTrigger -Daily -At "07:00AM"

$task1_settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

Register-ScheduledTask `
    -TaskName   $task1_name `
    -Action     $task1_action `
    -Trigger    $task1_trigger `
    -Settings   $task1_settings `
    -RunLevel   Limited `
    -Force

Write-Host "✓ Registered: $task1_name  (daily 7:00 AM)" -ForegroundColor Green

# ── Task 2: Weekly retrain every Sunday at 3:00 AM ───────────────────────────
$task2_name   = "Hogan_WeeklyRetrain"
$task2_script = "$HOGAN_ROOT\scripts\weekly_retrain.ps1"
$task2_action = New-ScheduledTaskAction `
    -Execute $PS_EXE `
    -Argument "-NonInteractive -File `"$task2_script`"" `
    -WorkingDirectory $HOGAN_ROOT

$task2_trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "03:00AM"

$task2_settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 15) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

Register-ScheduledTask `
    -TaskName   $task2_name `
    -Action     $task2_action `
    -Trigger    $task2_trigger `
    -Settings   $task2_settings `
    -RunLevel   Limited `
    -Force

Write-Host "✓ Registered: $task2_name  (weekly Sunday 3:00 AM)" -ForegroundColor Green

Write-Host ""
Write-Host "All tasks registered. Verify in Task Scheduler (taskschd.msc)." -ForegroundColor Cyan
Write-Host "Logs will appear in: $HOGAN_ROOT\logs\" -ForegroundColor Cyan

# ── Verify ───────────────────────────────────────────────────────────────────
Get-ScheduledTask | Where-Object { $_.TaskName -like "Hogan_*" } |
    Select-Object TaskName, State |
    Format-Table -AutoSize
