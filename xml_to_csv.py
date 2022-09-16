
import xml.etree.ElementTree as ET
import os
import csv

# Intended CSV format
# COLUMNS
# title stance arg1 arg2
from bs4 import BeautifulSoup

# Class to load data and manipulate it (process, transform, etc)
class DataLoad:
    # Initialize attribute self.data to hold 1 page raw xml data
    # Initialize attribute self.processed to hold all processed data
    def __init__(self):
        self.data = ''
        self.annotated_pairs = []
        self.expanded_pairs = []
        self.processed = []
        self.root = ''

    # Load the raw data in xml format and append to
    def load_data(self, data_file):
        with open(data_file, 'r', encoding='UTF-8') as f:
            self.data = f.read()
            self.root = ET.fromstring(self.data)

    # Parse the xml files and organize them into their respective annotated pairs
    def parse_xml(self):
        holder = self.root.findall('annotatedArgumentPair')
        for pair in holder:
            self.annotated_pairs.append(ET.tostring(pair))
        # print(len(self.annotated_pairs))

    # Expand pairs to their categories
    def expand_pairs(self):
        for i in range(len(self.annotated_pairs)):
            # print(len(self.annotated_pairs[i]))
            expanded = []
            for child in self.annotated_pairs[i]:
                # print(child.text)
                expanded.append(child)
            self.expanded_pairs.append(expanded)

    def parse_pairs(self):
        for pair in self.annotated_pairs:
            text = BeautifulSoup(pair, 'lxml-xml')
            title, stance = text.find('title').text, text.find('stance').text
            topic = title + stance
            arg1, arg2 = text.find_all('text')
            values, reasons = text.find_all('value'), text.find_all('reason')
            for i in range(len(values)):
                element = []
                element.extend([topic, arg1.text + '[SEP]' + arg2.text, values[i].text, reasons[i].text])
                self.processed.append(element)


# temporary single file for testing; will eventually need to iterate over all XML files
# should form a loop where there is a load - parse cycle. load 1 page and parse.
directory = 'UKPConvArg1/data/UKPConvArg1Strict-XML'
with open('UKPStrict.csv', 'w', encoding='UTF-8') as out:
    writer = csv.writer(out)
    writer.writerow(['topic', 'args', 'value', 'reason'])
    for filename in os.listdir(directory):
        my_data = DataLoad()
        my_data.load_data(os.path.join(directory, filename))

        my_data.parse_xml()
        my_data.parse_pairs()
        for data in my_data.processed:
            writer.writerow(data)

import xml.etree.ElementTree as ET
import os
import csv

# Intended CSV format
# COLUMNS
# title stance arg1 arg2
from bs4 import BeautifulSoup

# Class to load data and manipulate it (process, transform, etc)
class DataLoad:
    # Initialize attribute self.data to hold 1 page raw xml data
    # Initialize attribute self.processed to hold all processed data
    def __init__(self):
        self.data = ''
        self.annotated_pairs = []
        self.expanded_pairs = []
        self.processed = []
        self.root = ''

    # Load the raw data in xml format and append to
    def load_data(self, data_file):
        with open(data_file, 'r', encoding='UTF-8') as f:
            self.data = f.read()
            self.root = ET.fromstring(self.data)

    # Parse the xml files and organize them into their respective annotated pairs
    def parse_xml(self):
        holder = self.root.findall('annotatedArgumentPair')
        for pair in holder:
            self.annotated_pairs.append(ET.tostring(pair))
        # print(len(self.annotated_pairs))

    # Expand pairs to their categories
    def expand_pairs(self):
        for i in range(len(self.annotated_pairs)):
            # print(len(self.annotated_pairs[i]))
            expanded = []
            for child in self.annotated_pairs[i]:
                # print(child.text)
                expanded.append(child)
            self.expanded_pairs.append(expanded)

    def parse_pairs(self):
        for pair in self.annotated_pairs:
            text = BeautifulSoup(pair, 'lxml-xml')
            title, stance = text.find('title').text, text.find('stance').text
            topic = title + stance
            arg1, arg2 = text.find_all('text')
            values, reasons = text.find_all('value'), text.find_all('reason')
            for i in range(len(values)):
                element = []
                element.extend([topic, arg1.text + '[SEP]' + arg2.text, values[i].text, reasons[i].text])
                self.processed.append(element)


# temporary single file for testing; will eventually need to iterate over all XML files
# should form a loop where there is a load - parse cycle. load 1 page and parse.
directory = 'UKPConvArg1/data/UKPConvArg1Strict-XML'
with open('UKPStrict.csv', 'w', encoding='UTF-8') as out:
    writer = csv.writer(out)
    writer.writerow(['topic', 'args', 'value', 'reason'])
    for filename in os.listdir(directory):
        my_data = DataLoad()
        my_data.load_data(os.path.join(directory, filename))

        my_data.parse_xml()
        my_data.parse_pairs()
        for data in my_data.processed:
            writer.writerow(data)

import xml.etree.ElementTree as ET
import os
import csv

# Intended CSV format
# COLUMNS
# title stance arg1 arg2
from bs4 import BeautifulSoup

# Class to load data and manipulate it (process, transform, etc)
class DataLoad:
    # Initialize attribute self.data to hold 1 page raw xml data
    # Initialize attribute self.processed to hold all processed data
    def __init__(self):
        self.data = ''
        self.annotated_pairs = []
        self.expanded_pairs = []
        self.processed = []
        self.root = ''

    # Load the raw data in xml format and append to
    def load_data(self, data_file):
        with open(data_file, 'r', encoding='UTF-8') as f:
            self.data = f.read()
            self.root = ET.fromstring(self.data)

    # Parse the xml files and organize them into their respective annotated pairs
    def parse_xml(self):
        holder = self.root.findall('annotatedArgumentPair')
        for pair in holder:
            self.annotated_pairs.append(ET.tostring(pair))
        # print(len(self.annotated_pairs))

    # Expand pairs to their categories
    def expand_pairs(self):
        for i in range(len(self.annotated_pairs)):
            # print(len(self.annotated_pairs[i]))
            expanded = []
            for child in self.annotated_pairs[i]:
                # print(child.text)
                expanded.append(child)
            self.expanded_pairs.append(expanded)

    def parse_pairs(self):
        for pair in self.annotated_pairs:
            text = BeautifulSoup(pair, 'lxml-xml')
            title, stance = text.find('title').text, text.find('stance').text
            topic = title + stance
            arg1, arg2 = text.find_all('text')
            values, reasons = text.find_all('value'), text.find_all('reason')
            for i in range(len(values)):
                element = []
                element.extend([topic, arg1.text + '[SEP]' + arg2.text, values[i].text, reasons[i].text])
                self.processed.append(element)


# temporary single file for testing; will eventually need to iterate over all XML files
# should form a loop where there is a load - parse cycle. load 1 page and parse.
directory = 'UKPConvArg1/data/UKPConvArg1Strict-XML'
with open('UKPStrict.csv', 'w', encoding='UTF-8') as out:
    writer = csv.writer(out)
    writer.writerow(['topic', 'args', 'value', 'reason'])
    for filename in os.listdir(directory):
        my_data = DataLoad()
        my_data.load_data(os.path.join(directory, filename))

        my_data.parse_xml()
        my_data.parse_pairs()
        for data in my_data.processed:
            writer.writerow(data)

import xml.etree.ElementTree as ET
import os
import csv

# Intended CSV format
# COLUMNS
# title stance arg1 arg2
from bs4 import BeautifulSoup

# Class to load data and manipulate it (process, transform, etc)
class DataLoad:
    # Initialize attribute self.data to hold 1 page raw xml data
    # Initialize attribute self.processed to hold all processed data
    def __init__(self):
        self.data = ''
        self.annotated_pairs = []
        self.expanded_pairs = []
        self.processed = []
        self.root = ''

    # Load the raw data in xml format and append to
    def load_data(self, data_file):
        with open(data_file, 'r', encoding='UTF-8') as f:
            self.data = f.read()
            self.root = ET.fromstring(self.data)

    # Parse the xml files and organize them into their respective annotated pairs
    def parse_xml(self):
        holder = self.root.findall('annotatedArgumentPair')
        for pair in holder:
            self.annotated_pairs.append(ET.tostring(pair))
        # print(len(self.annotated_pairs))

    # Expand pairs to their categories
    def expand_pairs(self):
        for i in range(len(self.annotated_pairs)):
            # print(len(self.annotated_pairs[i]))
            expanded = []
            for child in self.annotated_pairs[i]:
                # print(child.text)
                expanded.append(child)
            self.expanded_pairs.append(expanded)

    def parse_pairs(self):
        for pair in self.annotated_pairs:
            text = BeautifulSoup(pair, 'lxml-xml')
            title, stance = text.find('title').text, text.find('stance').text
            topic = title + stance
            arg1, arg2 = text.find_all('text')
            values, reasons = text.find_all('value'), text.find_all('reason')
            for i in range(len(values)):
                element = []
                element.extend([topic, arg1.text + '[SEP]' + arg2.text, values[i].text, reasons[i].text])
                self.processed.append(element)


# temporary single file for testing; will eventually need to iterate over all XML files
# should form a loop where there is a load - parse cycle. load 1 page and parse.
directory = 'UKPConvArg1/data/UKPConvArg1Strict-XML'
with open('UKPStrict.csv', 'w', encoding='UTF-8') as out:
    writer = csv.writer(out)
    writer.writerow(['topic', 'args', 'value', 'reason'])
    for filename in os.listdir(directory):
        my_data = DataLoad()
        my_data.load_data(os.path.join(directory, filename))

        my_data.parse_xml()
        my_data.parse_pairs()
        for data in my_data.processed:
            writer.writerow(data)

import xml.etree.ElementTree as ET
import os
import csv

# Intended CSV format
# COLUMNS
# title stance arg1 arg2
from bs4 import BeautifulSoup

# Class to load data and manipulate it (process, transform, etc)
class DataLoad:
    # Initialize attribute self.data to hold 1 page raw xml data
    # Initialize attribute self.processed to hold all processed data
    def __init__(self):
        self.data = ''
        self.annotated_pairs = []
        self.expanded_pairs = []
        self.processed = []
        self.root = ''

    # Load the raw data in xml format and append to
    def load_data(self, data_file):
        with open(data_file, 'r', encoding='UTF-8') as f:
            self.data = f.read()
            self.root = ET.fromstring(self.data)

    # Parse the xml files and organize them into their respective annotated pairs
    def parse_xml(self):
        holder = self.root.findall('annotatedArgumentPair')
        for pair in holder:
            self.annotated_pairs.append(ET.tostring(pair))
        # print(len(self.annotated_pairs))

    # Expand pairs to their categories
    def expand_pairs(self):
        for i in range(len(self.annotated_pairs)):
            # print(len(self.annotated_pairs[i]))
            expanded = []
            for child in self.annotated_pairs[i]:
                # print(child.text)
                expanded.append(child)
            self.expanded_pairs.append(expanded)

    def parse_pairs(self):
        for pair in self.annotated_pairs:
            text = BeautifulSoup(pair, 'lxml-xml')
            title, stance = text.find('title').text, text.find('stance').text
            topic = title + stance
            arg1, arg2 = text.find_all('text')
            values, reasons = text.find_all('value'), text.find_all('reason')
            for i in range(len(values)):
                element = []
                element.extend([topic, arg1.text + '[SEP]' + arg2.text, values[i].text, reasons[i].text])
                self.processed.append(element)


# temporary single file for testing; will eventually need to iterate over all XML files
# should form a loop where there is a load - parse cycle. load 1 page and parse.
directory = 'UKPConvArg1/data/UKPConvArg1Strict-XML'
with open('UKPStrict.csv', 'w', encoding='UTF-8') as out:
    writer = csv.writer(out)
    writer.writerow(['topic', 'args', 'value', 'reason'])
    for filename in os.listdir(directory):
        my_data = DataLoad()
        my_data.load_data(os.path.join(directory, filename))

        my_data.parse_xml()
        my_data.parse_pairs()
        for data in my_data.processed:
            writer.writerow(data)

import xml.etree.ElementTree as ET
import os
import csv

# Intended CSV format
# COLUMNS
# title stance arg1 arg2
from bs4 import BeautifulSoup

# Class to load data and manipulate it (process, transform, etc)
class DataLoad:
    # Initialize attribute self.data to hold 1 page raw xml data
    # Initialize attribute self.processed to hold all processed data
    def __init__(self):
        self.data = ''
        self.annotated_pairs = []
        self.expanded_pairs = []
        self.processed = []
        self.root = ''

    # Load the raw data in xml format and append to
    def load_data(self, data_file):
        with open(data_file, 'r', encoding='UTF-8') as f:
            self.data = f.read()
            self.root = ET.fromstring(self.data)

    # Parse the xml files and organize them into their respective annotated pairs
    def parse_xml(self):
        holder = self.root.findall('annotatedArgumentPair')
        for pair in holder:
            self.annotated_pairs.append(ET.tostring(pair))
        # print(len(self.annotated_pairs))

    # Expand pairs to their categories
    def expand_pairs(self):
        for i in range(len(self.annotated_pairs)):
            # print(len(self.annotated_pairs[i]))
            expanded = []
            for child in self.annotated_pairs[i]:
                # print(child.text)
                expanded.append(child)
            self.expanded_pairs.append(expanded)

    def parse_pairs(self):
        for pair in self.annotated_pairs:
            text = BeautifulSoup(pair, 'lxml-xml')
            title, stance = text.find('title').text, text.find('stance').text
            topic = title + stance
            arg1, arg2 = text.find_all('text')
            values, reasons = text.find_all('value'), text.find_all('reason')
            for i in range(len(values)):
                element = []
                element.extend([topic, arg1.text + '[SEP]' + arg2.text, values[i].text, reasons[i].text])
                self.processed.append(element)


# temporary single file for testing; will eventually need to iterate over all XML files
# should form a loop where there is a load - parse cycle. load 1 page and parse.
directory = 'UKPConvArg1/data/UKPConvArg1Strict-XML'
with open('UKPStrict.csv', 'w', encoding='UTF-8') as out:
    writer = csv.writer(out)
    writer.writerow(['topic', 'args', 'value', 'reason'])
    for filename in os.listdir(directory):
        my_data = DataLoad()
        my_data.load_data(os.path.join(directory, filename))

        my_data.parse_xml()
        my_data.parse_pairs()
        for data in my_data.processed:
            writer.writerow(data)

import xml.etree.ElementTree as ET
import os
import csv

# Intended CSV format
# COLUMNS
# title stance arg1 arg2
from bs4 import BeautifulSoup

# Class to load data and manipulate it (process, transform, etc)
class DataLoad:
    # Initialize attribute self.data to hold 1 page raw xml data
    # Initialize attribute self.processed to hold all processed data
    def __init__(self):
        self.data = ''
        self.annotated_pairs = []
        self.expanded_pairs = []
        self.processed = []
        self.root = ''

    # Load the raw data in xml format and append to
    def load_data(self, data_file):
        with open(data_file, 'r', encoding='UTF-8') as f:
            self.data = f.read()
            self.root = ET.fromstring(self.data)

    # Parse the xml files and organize them into their respective annotated pairs
    def parse_xml(self):
        holder = self.root.findall('annotatedArgumentPair')
        for pair in holder:
            self.annotated_pairs.append(ET.tostring(pair))
        # print(len(self.annotated_pairs))

    # Expand pairs to their categories
    def expand_pairs(self):
        for i in range(len(self.annotated_pairs)):
            # print(len(self.annotated_pairs[i]))
            expanded = []
            for child in self.annotated_pairs[i]:
                # print(child.text)
                expanded.append(child)
            self.expanded_pairs.append(expanded)

    def parse_pairs(self):
        for pair in self.annotated_pairs:
            text = BeautifulSoup(pair, 'lxml-xml')
            title, stance = text.find('title').text, text.find('stance').text
            topic = title + stance
            arg1, arg2 = text.find_all('text')
            values, reasons = text.find_all('value'), text.find_all('reason')
            for i in range(len(values)):
                element = []
                element.extend([topic, arg1.text + '[SEP]' + arg2.text, values[i].text, reasons[i].text])
                self.processed.append(element)


# temporary single file for testing; will eventually need to iterate over all XML files
# should form a loop where there is a load - parse cycle. load 1 page and parse.
directory = 'UKPConvArg1/data/UKPConvArg1Strict-XML'
with open('UKPStrict.csv', 'w', encoding='UTF-8') as out:
    writer = csv.writer(out)
    writer.writerow(['topic', 'args', 'value', 'reason'])
    for filename in os.listdir(directory):
        my_data = DataLoad()
        my_data.load_data(os.path.join(directory, filename))

        my_data.parse_xml()
        my_data.parse_pairs()
        for data in my_data.processed:
            writer.writerow(data)
