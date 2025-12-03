import { ApiProperty } from '@nestjs/swagger';

export class ReceiptResponseDto {
  @ApiProperty()
  id: number;

  @ApiProperty()
  memberId: string;

  @ApiProperty({ required: false })
  placeId?: number;

  @ApiProperty()
  productDescription: string;

  @ApiProperty()
  photoPath: string;

  @ApiProperty()
  pointsEarned: number;

  @ApiProperty({ enum: ['PENDING', 'APPROVED', 'REJECTED'] })
  status: string;

  @ApiProperty({ required: false })
  verificationResult?: string;

  @ApiProperty()
  createdAt: Date;

  @ApiProperty()
  updatedAt: Date;
}
