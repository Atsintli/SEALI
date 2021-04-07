from csv import writer
from csv import reader
def add_column_in_csv(input_file, output_file, transform_row):
    """ Append a column in existing csv using csv.reader / csv.writer classes"""
    # Open the input_file in read mode and output_file in write mode
    with open(input_file, 'r') as read_obj, \
            open(output_file, 'w', newline='') as write_obj:
        # Create a csv.reader object from the input file object
        csv_reader = reader(read_obj)
        # Create a csv.writer object from the output file object
        csv_writer = writer(write_obj)
        # Read each row of the input csv file as list
        for row in csv_reader:
            # Pass the list / row in the transform function to add column text for this row
            transform_row(row, csv_reader.line_num)
            # Write the updated row / list to the output file
            csv_writer.writerow(row)

file_in = "sink_into_return_test.csv"
list_of_str = ['First', 'Second', 'Third', 'Fourth', 'Fifth','First', 'Second', 'Third', 'Fourth', 'Fifth']
add_column_in_csv(file_in, 'output_4.csv', lambda row, line_num: row.append(list_of_str[line_num -1]))