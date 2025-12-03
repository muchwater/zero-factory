import {
  Controller,
  Post,
  Get,
  Param,
  Query,
  Body,
  UseInterceptors,
  UploadedFile,
  ParseIntPipe,
} from '@nestjs/common';
import {
  ApiTags,
  ApiOperation,
  ApiResponse,
  ApiConsumes,
  ApiBody,
} from '@nestjs/swagger';
import { FileInterceptor } from '@nestjs/platform-express';
import { ReceiptsService } from './receipts.service';
import { CreateReceiptDto } from './dto/create-receipt.dto';
import { ReceiptResponseDto } from './dto/receipt-response.dto';
import { GetReceiptsQueryDto } from './dto/get-receipts-query.dto';
import { ReceiptHistoryResponseDto } from './dto/receipt-history-response.dto';

@ApiTags('receipts')
@Controller()
export class ReceiptsController {
  constructor(private readonly receiptsService: ReceiptsService) {}

  /**
   * 영수증 제출
   */
  @Post('members/:memberId/receipts')
  @UseInterceptors(FileInterceptor('photo'))
  @ApiOperation({
    summary: '영수증 제출',
    description: '재사용 용기 사용 영수증을 제출하고 100포인트를 적립합니다.',
  })
  @ApiConsumes('multipart/form-data')
  @ApiBody({
    schema: {
      type: 'object',
      properties: {
        photo: {
          type: 'string',
          format: 'binary',
          description: '영수증 사진',
        },
        productDescription: {
          type: 'string',
          description: '제품/서비스 설명',
          example: 'One Americano',
        },
        placeId: {
          type: 'number',
          description: '장소 ID (선택)',
        },
        verificationResult: {
          type: 'string',
          description: 'AI 검증 결과 JSON (선택)',
        },
      },
      required: ['photo', 'productDescription'],
    },
  })
  @ApiResponse({
    status: 201,
    description: '영수증 제출 성공',
    type: ReceiptResponseDto,
  })
  async submitReceipt(
    @Param('memberId') memberId: string,
    @Body() dto: CreateReceiptDto,
    @UploadedFile() file: Express.Multer.File,
  ): Promise<ReceiptResponseDto> {
    console.log('📝 Receipt submission request:', {
      memberId,
      dto,
      hasFile: !!file,
      fileSize: file?.size,
      fileName: file?.originalname,
    });
    return this.receiptsService.create(memberId, dto, file);
  }

  /**
   * 제출 이력 조회
   */
  @Get('members/:memberId/receipts')
  @ApiOperation({
    summary: '제출 이력 조회',
    description: '회원의 영수증 제출 이력을 페이지네이션으로 조회합니다.',
  })
  @ApiResponse({
    status: 200,
    description: '조회 성공',
    type: ReceiptHistoryResponseDto,
  })
  async getSubmissionHistory(
    @Param('memberId') memberId: string,
    @Query() query: GetReceiptsQueryDto,
  ): Promise<ReceiptHistoryResponseDto> {
    return this.receiptsService.findByMember(memberId, query);
  }

  /**
   * 영수증 상세 조회
   */
  @Get('receipts/:receiptId')
  @ApiOperation({
    summary: '영수증 상세 조회',
    description: '특정 영수증의 상세 정보를 조회합니다.',
  })
  @ApiResponse({
    status: 200,
    description: '조회 성공',
    type: ReceiptResponseDto,
  })
  @ApiResponse({
    status: 404,
    description: '영수증을 찾을 수 없습니다.',
  })
  async getReceiptDetail(
    @Param('receiptId', ParseIntPipe) receiptId: number,
  ): Promise<ReceiptResponseDto> {
    return this.receiptsService.findById(receiptId);
  }
}
