## File Format
The .mag file format is a cross-platform binary format for storing models.
Tensors and metadata are both identified by UTF-8 strings, so the file content assembles two dictionary-like structures:
- A dictionary of tensors, where each tensor is identified by a unique key.
- A dictionary of metadata, where each metadata entry is identified by a unique key.
Tensor and metadata keys are distinct, meaning a tensor and a metadata entry can have the same key without conflict.
Memory mapping is supported and internal checksums are provided to ensure data integrity for headers and tensor data.

## Version
The format version is a continuous integer starting from 1.
The current version is 1.

## Data Layout
The file structure consists of a header, followed by metadata records, tensor records, and finally the tensor data buffer.
The checksum field in the header is a CRC32 checksum of the file header, metadata records, and tensor records but not the tensor data buffer.
To get the pointer to the tensor data buffer, the `data_offset` field in each tensor record is used.
The auxiliary field `aux` in the header and metadata records contain bit-packed data. Some bits are reserved for future use and must be zero.
Each tensor data buffer is stored in a contiguous block at the end of the file, and the offset to this buffer is calculated from the start of the file.
The start address is also always 64-byte aligned.

## Limits
* Max key length: 2^16 - 1 bytes
* Max number of tensors: 2^32 - 1
* Max number of metadata entries: 2^32 - 1
* Min tensor rank: 1
* Max tensor rank: 255
* Max tensor shape dimension: 2^64/2 - 1 elements
* Max tensor elements: 2^64/2 - 1 elements
* Max tensor data size: 2^64/2 - 1 bytes

### File Header
| Name        | Type  | Description                   |
|-------------|-------|-------------------------------|
| magic       | u8[4] | Magic number, always "MAG!"   |
| version     | u32   | Version of the file format    |
| checksum    | u32   | CRC32 of the file headers     |
| num_tensors | u32   | Number of tensors in the file |
| num_meta_kv | u32   | Number of metadata entries    |
| aux         | u32   | Auxiliary field, reserved     |

### Metadata Record
| Name             | Type           | Description                                                          |
|------------------|----------------|----------------------------------------------------------------------|
| aux              | u32            | Auxiliary packed field. Contents: (type: u8). Other bits reserved.   |
| payload          | u64            | Payload bits                                                         |
| ---------------- | -------------  | -------------------------------------------------------------------- |
| key_length       | u32            | Length of the key string                                             |
| key              | u8[key_length] | UTF-8 string, not null terminated                                    |

### Tensor Record
| Name             | Type           | Description                                                                     |
|------------------|----------------|---------------------------------------------------------------------------------|
| aux              | u32            | Auxiliary packed field. Contents: (dtype: u8, rank: u8). Other bits reserved.   |
| numel            | u64            | Number of elements                                                              |
| offset           | u64            | Absolute offset of the data buffer to the file start                            |
| ---------------- | -------------  | ------------------------------------------------------------------------------- |
| shape            | u64[rank]      | Shape of the tensor, rank elements                                              |
| ---------------- | -------------  | ------------------------------------------------------------------------------- |
| key_length       | u32            | Length of the key string                                                        |
| key              | u8[key_length] | UTF-8 string, not null terminated                                               |

## File Structure
| Name                           | Included in file checksum? |
|--------------------------------|----------------------------|
| File Header                    | Yes                        |
| Metadata Records[num_metadata] | Yes                        |
| Tensor Record[num_tensors]     | Yes                        |
| Tensor Data Buffer             | No                         |