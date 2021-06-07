# Dataset

## Directory structure

Format of the filename is as follows:

```
{server}/{User Agent}/{website}/{timestamp}_{browser}.summary
```

## File format

Each file is a CSV that can be easily load with `pd.read_csv`. Columns are:
- serverIP
- serverPort
- proto (IP numbers, https://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml)
- totalBytes
- totalPackets
- srcBytes
- dstBytes
- srcPackets
- dstPackets

## More?

If you want to obtain more detailed version of the dataset in PCAP format, you can contact me (daniel.perdices at uam.es) and we can agree a way of sending it.
