#open multiple workbooks
Sub macroopen()
Const sPath = "G:\global\china forecasting service\Data\Prefectures\Demographics\"
sfil = Dir(sPath & "CNHN*.xls*")
i = 1
Do While sfil <> "" And i < 15 '15 is the max number of files to be open!!!otherwise files won't be savevd
Workbooks.Open sPath & sfil, "0"
sfil = Dir
i = i + 1
Loop
End Sub

#use solver to feed the forecasts into the workbooks
Sub macromuni_c()
'to facilitate column operations
Workbooks("Prefecture_Central").Activate
Sheets("MigrantsAllocation").Activate
For n = 26 To 26
    For m = 1029 To 1029
        j = Application.WorksheetFunction.Match(Cells(m, 4).Value, Range("D1:D296"), 0)
        'If Not IsEmpty(Cells(m, n).Offset(0, 30)) Then
        If Abs(Cells(m, n).Offset(0, 30)) > 0.0001 Then
        Application.Run "SolverReset"
        'To allow for negative multipliers
        Application.Run "SolverOptions", "100", "100", "0.0001", "False", "False", "1", "1", "1", "5", "False", "0.0001", "False"
        Application.Run "SolverOk", Cells(m, n).Offset(0, 30), "3", "0", Union(Cells(j, n), Cells(j, n).Offset(296, 0))
        'Application.Run "SolverAdd", Cells(m, n).Offset(1, 0), "3", Cells(m, n).Offset(1, -1)
        Application.Run "SolverSolve", True
        'End If
        End If
    Next m
Next n
End Sub

#use goseek to feed the forecasts into the workbooks
Sub macroseek()
'to facilitate row operations
Workbooks("Prefecture_Central").Activate
Sheets("MigrantsAllocation").Activate
Dim n As Integer
Dim m As Integer
j = Range("A1", "BD1").Columns.Count - Range("A1", "Z1").Columns.Count
For m = 944 To 950
    i = Application.WorksheetFunction.Match(Cells(m, 4).Value, Range("D1:D296"), 0)
    For n = 33 To 40
        If Cells(m, n).GoalSeek(Goal:=Cells(m, n).Offset(0, j), ChangingCell:=Cells(i, n)) = False Then
            Exit For
        End If
    Next n
Next m
End Sub

#save all the changes made to the workbooks
Sub macrosave()
Dim wb As Workbook
For Each wb In Application.Workbooks
    If wb.Name <> "Prefecture_Central.xlsx" And wb.Name <> "ChinaRoller_1.xls" Then
        wb.Close SaveChanges:=True
    End If
Next wb
End Sub

#for second check
Sub macrocolor1()
Workbooks("Prefecture_Central").Activate
Sheets("Birth").Activate
m = Range("b1", ActiveCell).Rows.Count
n = Range("b1", ActiveCell.End(xlDown)).Rows.Count
For i = m To n
    If Cells(i, "V") >= Cells(i, "R") Then
        Cells(i, "B").EntireRow.Interior.ColorIndex = 6
    End If
Next i
End Sub

#for second check
Sub macrocolor2()
Workbooks("Prefecture_Central").Activate
Sheets("Population_Total").Activate
m = Range("b1", ActiveCell).Rows.Count
n = Range("b1", ActiveCell.End(xlDown)).Rows.Count
For i = m To n
    If Abs(Cells(i, "BY") - Cells(i, "BZ")) > 20 Then
        Cells(i, "B").EntireRow.Interior.ColorIndex = 6
    End If
Next i
End Sub
