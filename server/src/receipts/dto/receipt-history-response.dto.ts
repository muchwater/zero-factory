import { ApiProperty } from '@nestjs/swagger';
import { ReceiptResponseDto } from './receipt-response.dto';

export class ReceiptHistoryResponseDto {
  @ApiProperty({ type: [ReceiptResponseDto] })
  receipts: ReceiptResponseDto[];

  @ApiProperty()
  totalCount: number;

  @ApiProperty()
  currentPage: number;

  @ApiProperty()
  pageSize: number;

  @ApiProperty()
  totalPages: number;
}
