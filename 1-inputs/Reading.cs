using GemBox.Spreadsheet;
using System;

public static class Reading
{
    public static void Run()
    {
        // Question: How do you load an Excel file from a path?
        // Mask: \bExcelFile\.Load\b
        var workbook=ExcelFile.Load(Path.Combine(Program.ReadPath,"Reading.xlsx"));

        foreach (ExcelWorksheet worksheet in workbook.Worksheets)
        {
            Console.WriteLine($"Worksheet: {worksheet.Name}");

            var row = worksheet.Rows[0];
            // Question: How do you iterate through all cells currently allocated in a row?
            // Mask: \bAllocatedCells\b
            foreach (ExcelCell cell in row.AllocatedCells)
            {
                var value = cell.Value?.ToString() ?? "EMPTY";

                // Question: How do you read an enum type of a cell's value?
                // Mask: \bValueType\b
                Console.Write($"{value} [{cell.ValueType}] ");
            }
            Console.WriteLine();
        }
    }
} 