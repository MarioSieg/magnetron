## File Format
The .mag file format is a cross-platform binary format for storing networks and tensors.
The format allows to store tensors identified by a string key and metadata of various data types.
Memory mapping support allows for very fast loading of large files.
All data is always stored in little endian format. If the host machine is big endian, the data will be converted to little endian format on load.

## Data Layout
The file structure consists of a header, followed by metadata records, tensor records, and finally the tensor data buffer.
The checksum field in the header is a CRC32 checksum of the file header, metadata records, and tensor records but not the tensor data buffer.
To get the pointer to the tensor data buffer, the `data_offset` field in each tensor record is used.
The auxiliary field `aux` in the header and metadata records are reserved for future use and should be set to zero when writing files.
### File Header
| Name        | Type       | Description                   |
|-------------|------------|-------------------------------|
| magic       | uint8_t[4] | Magic number, always "MAG!"   |
| version     | uint32_t   | Version of the file format    |
| checksum    | uint32_t   | CRC32 of the file headers     |
| num_tensors | uint32_t   | Number of tensors in the file |
| num_meta_kv | uint32_t   | Number of metadata entries    |
| aux         | uint32_t   | Auxiliary field, reserved     |

### Metadata Record
| Name           | Type        | Description                         |
|----------------|-------------|-------------------------------------|
| aux            | uint32_t    | Auxiliary field                     |
| payload        | uint64_t    | Payload bits                        |
| -------------- | ----------- | ----------------------------------- |
| key_length     | uint32_t    | Length of the key string            |
| key            | uint8_t[]   | UTF-8 string, not null terminated   |

### Tensor Record
| Name        | Type       | Description                             |
|-------------|------------|-----------------------------------------|
| key_length  | uint32_t   | Length of the key string                |
| key         | uint8_t[]  | UTF-8 string, not null terminated       |
| shape       | int64_t[6] | Shape of the tensor, up to 6 dimensions |
| dtype       | uint8_t    | Data type of the tensor                 |
| data_length | uint64_t   | Length of the data in bytes             |
| data_offset | uint64_t   | Offset of the data in the file          |
| aux         | uint32_t   | Auxiliary field, reserved               |

## File Structure
| Name                                  | Included in file checksum? |
|---------------------------------------|----------------------------|
| File Header                           | Yes                        |
| Metadata Records (num_metadata times) | Yes                        |
| Tensor Records (num_tensors times)    | Yes                        |
| Tensor Data Buffer                    | No                         |