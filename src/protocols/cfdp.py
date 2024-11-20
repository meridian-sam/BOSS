from enum import Enum, auto
from dataclasses import dataclass
import struct
from typing import Optional, Dict, List, Tuple
import logging
from datetime import datetime
import hashlib

class CFDPMessageType(Enum):
    """CFDP PDU types."""
    FILE_DIRECTIVE = 0
    FILE_DATA = 1

class DirectiveCode(Enum):
    """CFDP directive codes."""
    METADATA = 1
    EOF = 4
    FINISHED = 5
    ACK = 6
    NAK = 7
    PROMPT = 8
    KEEP_ALIVE = 9
    ERROR = 15

class TransactionStatus(Enum):
    """CFDP transaction states."""
    IDLE = auto()
    SENDING = auto()
    RECEIVING = auto()
    PAUSED = auto()
    FAILED = auto()
    COMPLETED = auto()

class ConditionCode(Enum):
    """CFDP condition codes."""
    NO_ERROR = 0
    POSITIVE_ACK_LIMIT_REACHED = 1
    KEEP_ALIVE_LIMIT_REACHED = 2
    INVALID_TRANSMISSION_MODE = 3
    FILESTORE_REJECTION = 4
    FILE_CHECKSUM_FAILURE = 5
    FILE_SIZE_ERROR = 6
    NAK_LIMIT_REACHED = 7
    INACTIVITY_DETECTED = 8
    CHECK_LIMIT_REACHED = 9
    UNSUPPORTED_CHECKSUM_TYPE = 10

@dataclass
class CFDPHeader:
    """CFDP PDU header."""
    version: int = 1
    pdu_type: CFDPMessageType = CFDPMessageType.FILE_DATA
    direction: int = 0  # 0: toward receiver, 1: toward sender
    transmission_mode: int = 0  # 0: acknowledged, 1: unacknowledged
    crc_flag: bool = True
    file_size: Optional[int] = None
    source_id: int = 0
    destination_id: int = 0
    sequence_number: int = 0

@dataclass
class CFDPTransaction:
    """CFDP transaction state."""
    transaction_id: int
    source_id: int
    destination_id: int
    filename: str
    file_size: int
    status: TransactionStatus
    start_time: datetime
    segments_sent: List[int]
    segments_received: List[int]
    checksum: Optional[bytes] = None
    condition_code: ConditionCode = ConditionCode.NO_ERROR
    transmission_mode: int = 0
    retry_count: int = 0
    max_retries: int = 3
    last_activity: datetime = datetime.utcnow()

class CFDPProtocol:
    """CCSDS File Delivery Protocol implementation."""
    
    def __init__(self, entity_id: int, segment_size: int = 1024):
        """
        Initialize CFDP protocol handler.

        Args:
            entity_id: Unique identifier for this CFDP entity
            segment_size: Maximum size of file data segments
        """
        self.entity_id = entity_id
        self.segment_size = segment_size
        self.transaction_seq = 0
        self.active_transactions: Dict[int, CFDPTransaction] = {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.keep_alive_interval = 60  # seconds
        self.inactivity_timeout = 300  # seconds
        self.ack_timeout = 10  # seconds
        
    def create_transaction(self, filename: str, file_size: int, 
                         destination_id: int, transmission_mode: int = 0) -> CFDPTransaction:
        """
        Create new CFDP transaction.

        Args:
            filename: Name of file to transfer
            file_size: Size of file in bytes
            destination_id: Destination entity ID
            transmission_mode: 0 for acknowledged, 1 for unacknowledged

        Returns:
            New transaction object
        """
        transaction = CFDPTransaction(
            transaction_id=self.transaction_seq,
            source_id=self.entity_id,
            destination_id=destination_id,
            filename=filename,
            file_size=file_size,
            status=TransactionStatus.IDLE,
            start_time=datetime.utcnow(),
            segments_sent=[],
            segments_received=[],
            transmission_mode=transmission_mode
        )
        self.active_transactions[self.transaction_seq] = transaction
        self.transaction_seq += 1
        return transaction

    def create_metadata_pdu(self, transaction: CFDPTransaction) -> bytes:
        """Create metadata PDU for file transfer initiation."""
        header = CFDPHeader(
            pdu_type=CFDPMessageType.FILE_DIRECTIVE,
            file_size=transaction.file_size,
            source_id=self.entity_id,
            destination_id=transaction.destination_id,
            sequence_number=transaction.transaction_id,
            transmission_mode=transaction.transmission_mode
        )
        
        # Pack header
        header_bytes = self._pack_header(header)
        
        # Pack metadata directive
        metadata = struct.pack(
            f">BIH{len(transaction.filename)}s",
            DirectiveCode.METADATA.value,
            transaction.file_size,
            len(transaction.filename),
            transaction.filename.encode()
        )
        
        return header_bytes + metadata
    
    def create_file_data_pdu(self, transaction: CFDPTransaction, 
                            offset: int, data: bytes) -> bytes:
        """Create file data PDU."""
        header = CFDPHeader(
            pdu_type=CFDPMessageType.FILE_DATA,
            source_id=self.entity_id,
            destination_id=transaction.destination_id,
            sequence_number=transaction.transaction_id
        )
        
        header_bytes = self._pack_header(header)
        
        # Pack offset and data
        segment = struct.pack(">I", offset) + data
        
        return header_bytes + segment
    
    def create_eof_pdu(self, transaction: CFDPTransaction) -> bytes:
        """Create EOF PDU."""
        header = CFDPHeader(
            pdu_type=CFDPMessageType.FILE_DIRECTIVE,
            source_id=self.entity_id,
            destination_id=transaction.destination_id,
            sequence_number=transaction.transaction_id
        )
        
        header_bytes = self._pack_header(header)
        
        # Pack EOF directive
        eof_directive = struct.pack(
            ">BIHH16s",
            DirectiveCode.EOF.value,
            transaction.file_size,
            transaction.condition_code.value,
            0,  # Checksum type (0 = CRC32)
            transaction.checksum or b'\x00' * 16
        )
        
        return header_bytes + eof_directive

    def create_ack_pdu(self, transaction: CFDPTransaction, 
                      directive_code: DirectiveCode) -> bytes:
        """Create acknowledgment PDU."""
        header = CFDPHeader(
            pdu_type=CFDPMessageType.FILE_DIRECTIVE,
            direction=1,  # toward sender
            source_id=self.entity_id,
            destination_id=transaction.destination_id,
            sequence_number=transaction.transaction_id
        )
        
        header_bytes = self._pack_header(header)
        
        # Pack ACK directive
        ack_directive = struct.pack(
            ">BBH",
            DirectiveCode.ACK.value,
            directive_code.value,
            transaction.condition_code.value
        )
        
        return header_bytes + ack_directive

    def create_nak_pdu(self, transaction: CFDPTransaction, 
                      start_offset: int, end_offset: int) -> bytes:
        """Create NAK (negative acknowledgment) PDU."""
        header = CFDPHeader(
            pdu_type=CFDPMessageType.FILE_DIRECTIVE,
            direction=1,  # toward sender
            source_id=self.entity_id,
            destination_id=transaction.destination_id,
            sequence_number=transaction.transaction_id
        )
        
        header_bytes = self._pack_header(header)
        
        # Pack NAK directive
        nak_directive = struct.pack(
            ">BII",
            DirectiveCode.NAK.value,
            start_offset,
            end_offset
        )
        
        return header_bytes + nak_directive

    def process_pdu(self, pdu: bytes) -> Tuple[CFDPHeader, bytes]:
        """
        Process received PDU.

        Returns:
            Tuple of (header, data)
        """
        header, data = self._unpack_header(pdu)
        
        # Update transaction status
        if header.sequence_number in self.active_transactions:
            transaction = self.active_transactions[header.sequence_number]
            transaction.last_activity = datetime.utcnow()
            
            if header.pdu_type == CFDPMessageType.FILE_DIRECTIVE:
                self._process_directive(transaction, data)
            else:
                self._process_file_data(transaction, data)
        
        return header, data

    def _pack_header(self, header: CFDPHeader) -> bytes:
        """Pack CFDP header into bytes."""
        first_byte = (
            (header.version & 0x03) << 5 |
            (header.pdu_type.value & 0x01) << 4 |
            (header.direction & 0x01) << 3 |
            (header.transmission_mode & 0x01) << 2 |
            (header.crc_flag & 0x01) << 1
        )
        
        return struct.pack(
            ">BHHI",
            first_byte,
            header.source_id,
            header.destination_id,
            header.sequence_number
        )

    def _unpack_header(self, data: bytes) -> Tuple[CFDPHeader, bytes]:
        """Unpack CFDP header from bytes."""
        if len(data) < 8:
            raise ValueError("Insufficient data for CFDP header")
            
        first_byte, source_id, dest_id, seq_num = struct.unpack(">BHHI", data[:8])
        
        header = CFDPHeader(
            version=(first_byte >> 5) & 0x03,
            pdu_type=CFDPMessageType((first_byte >> 4) & 0x01),
            direction=(first_byte >> 3) & 0x01,
            transmission_mode=(first_byte >> 2) & 0x01,
            crc_flag=bool((first_byte >> 1) & 0x01),
            source_id=source_id,
            destination_id=dest_id,
            sequence_number=seq_num
        )
        
        return header, data[8:]

    def _process_directive(self, transaction: CFDPTransaction, data: bytes):
        """Process directive PDU."""
        directive_code = DirectiveCode(data[0])
        
        if directive_code == DirectiveCode.EOF:
            file_size, condition_code, _, checksum = struct.unpack(">IHH16s", data[1:25])
            transaction.checksum = checksum
            transaction.condition_code = ConditionCode(condition_code)
            
            if transaction.file_size == file_size and transaction.condition_code == ConditionCode.NO_ERROR:
                transaction.status = TransactionStatus.COMPLETED
            else:
                transaction.status = TransactionStatus.FAILED
                
        elif directive_code == DirectiveCode.ACK:
            acked_directive, condition_code = struct.unpack(">BH", data[1:4])
            if DirectiveCode(acked_directive) == DirectiveCode.EOF:
                transaction.status = TransactionStatus.COMPLETED
                
        elif directive_code == DirectiveCode.NAK:
            start_offset, end_offset = struct.unpack(">II", data[1:9])
            # Handle retransmission request
            transaction.segments_received = [i for i in range(start_offset, end_offset)]

    def _process_file_data(self, transaction: CFDPTransaction, data: bytes):
        """Process file data PDU."""
        offset = struct.unpack(">I", data[:4])[0]
        transaction.segments_received.append(offset)
        transaction.status = TransactionStatus.RECEIVING

    def update_transactions(self):
        """Update status of active transactions."""
        current_time = datetime.utcnow()
        
        for transaction in list(self.active_transactions.values()):
            # Check for inactivity timeout
            if (current_time - transaction.last_activity).total_seconds() > self.inactivity_timeout:
                transaction.status = TransactionStatus.FAILED
                transaction.condition_code = ConditionCode.INACTIVITY_DETECTED
                
            # Handle retries for acknowledged mode
            if (transaction.transmission_mode == 0 and 
                transaction.status == TransactionStatus.SENDING and
                transaction.retry_count < transaction.max_retries):
                
                # Check if acknowledgment timeout has occurred
                if (current_time - transaction.last_activity).total_seconds() > self.ack_timeout:
                    transaction.retry_count += 1
                    # Resend last segment
                    if transaction.segments_sent:
                        last_segment = transaction.segments_sent[-1]
                        # Implementation-specific resend logic here