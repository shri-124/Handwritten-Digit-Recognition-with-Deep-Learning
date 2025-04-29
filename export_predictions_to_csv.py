import pandas as pd
import sqlite3

# Connect to your database
conn = sqlite3.connect("mnist_predictions.db")

# Read the predictions table into a DataFrame
df = pd.read_sql_query("SELECT * FROM predictions", conn)

# Save the DataFrame to a CSV file
df.to_csv("mnist_predictions.csv", index=False)

# Close the connection
conn.close()

print("Predictions exported to mnist_predictions.csv!")
