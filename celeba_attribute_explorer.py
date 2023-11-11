import pandas as pd
import ipdb

def load_data_as_dataframe(filename):
    # Read the first line to get the attribute names
    with open(filename, 'r') as file:
        file.readline() # num entries
        attributes = file.readline().strip().split()
        print(attributes)

    # Use Pandas to read the data, skipping the first row which is the header
    df = pd.read_csv(filename, sep='\s+', names=attributes, skiprows=2)

    return df

def query_images(df, attribute, value):
    # Filter the dataframe based on the attribute and value
    filtered_df = df[df[attribute] == value]
    return filtered_df.index.tolist()  # Return the list of image names

# Usage
filename = 'CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt'  # Replace with your actual file name
df = load_data_as_dataframe(filename)

ipdb.set_trace()


# Example query for 'Male' attribute
male_images = query_images(df, 'Male', 1)
print("Male Images:", male_images)
