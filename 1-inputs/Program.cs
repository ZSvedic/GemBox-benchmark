using System;

public class Program
{
    public static readonly string ReadPath = "read/";
    public static readonly string WritePath = "write/";

    static void Main(string[] args)
    {
        Console.WriteLine($"Hello from .NET {System.Environment.Version}");

        Console.WriteLine("Running: Usage");
        Usage.Run();

        Console.WriteLine("Running: Formatting");
        Formatting.Run();

        Console.WriteLine("Running: Reading");
        Reading.Run();

        Console.WriteLine("Running: Charts");
        Charts.Run();

        Console.WriteLine("Running: PrintView");
        PrintView.Run();

        Console.WriteLine("Running: Writing");
        Writing.Run();
    }
}
