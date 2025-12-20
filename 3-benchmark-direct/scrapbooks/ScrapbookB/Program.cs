using System;
using GemBox.Spreadsheet;
class Program {
    static void Main() {
        SpreadsheetInfo.SetLicense("FREE-LIMITED-KEY");
        var ef = new ExcelFile();
        var ws = ef.Worksheets.Add("Sheet1").Cells[0, 0].Value = "Hello, GemBox!";
        ef.Save("Output.xlsx");
        Console.WriteLine("Hello")
    }
}