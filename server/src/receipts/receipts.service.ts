import {
  Injectable,
  NotFoundException,
  InternalServerErrorException,
} from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { PointsService } from '../points/points.service';
import { CreateReceiptDto } from './dto/create-receipt.dto';
import { ReceiptResponseDto } from './dto/receipt-response.dto';
import { GetReceiptsQueryDto } from './dto/get-receipts-query.dto';
import { ReceiptHistoryResponseDto } from './dto/receipt-history-response.dto';
import * as fs from 'fs';
import * as path from 'path';
import { v4 as uuidv4 } from 'uuid';

@Injectable()
export class ReceiptsService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly pointsService: PointsService,
  ) {}

  /**
   * 영수증 생성 및 자동 승인 처리
   */
  async create(
    memberId: string,
    dto: CreateReceiptDto,
    file: Express.Multer.File,
  ): Promise<ReceiptResponseDto> {
    try {
      // 1. 파일 저장
      const photoPath = await this.savePhoto(file, memberId);

      // 2. Receipt 생성 (자동 승인)
      const receipt = await this.prisma.receipt.create({
        data: {
          memberId,
          productDescription: dto.productDescription,
          photoPath,
          pointsEarned: 100,
          status: 'APPROVED',
          placeId: dto.placeId || undefined,
          verificationResult: dto.verificationResult || undefined,
        },
      });

      // 3. 포인트 적립
      await this.pointsService.earnPoints(memberId, 100, dto.placeId);

      return receipt as ReceiptResponseDto;
    } catch (error) {
      throw new InternalServerErrorException(
        '영수증 제출 처리 중 오류가 발생했습니다.',
      );
    }
  }

  /**
   * 회원별 영수증 조회 (페이지네이션)
   */
  async findByMember(
    memberId: string,
    query: GetReceiptsQueryDto,
  ): Promise<ReceiptHistoryResponseDto> {
    const { page = 1, limit = 20, status } = query;
    const skip = (page - 1) * limit;

    // 필터 조건 구성
    const where: any = { memberId };
    if (status) {
      where.status = status;
    }

    // 데이터 조회
    const [receipts, totalCount] = await Promise.all([
      this.prisma.receipt.findMany({
        where,
        orderBy: { createdAt: 'desc' },
        skip,
        take: limit,
      }),
      this.prisma.receipt.count({ where }),
    ]);

    const totalPages = Math.ceil(totalCount / limit);

    return {
      receipts: receipts as ReceiptResponseDto[],
      totalCount,
      currentPage: page,
      pageSize: limit,
      totalPages,
    };
  }

  /**
   * 영수증 상세 조회
   */
  async findById(receiptId: number): Promise<ReceiptResponseDto> {
    const receipt = await this.prisma.receipt.findUnique({
      where: { id: receiptId },
    });

    if (!receipt) {
      throw new NotFoundException(`Receipt with ID ${receiptId} not found`);
    }

    return receipt as ReceiptResponseDto;
  }

  /**
   * 파일 저장 헬퍼 메서드
   */
  private async savePhoto(
    file: Express.Multer.File,
    memberId: string,
  ): Promise<string> {
    try {
      // UUID 생성
      const uuid = uuidv4();
      const ext = path.extname(file.originalname);
      const filename = `${uuid}${ext}`;

      // 저장 디렉토리 생성
      const uploadDir = path.join(
        process.cwd(),
        'uploads',
        'receipts',
        memberId,
      );
      if (!fs.existsSync(uploadDir)) {
        fs.mkdirSync(uploadDir, { recursive: true });
      }

      // 파일 저장
      const filePath = path.join(uploadDir, filename);
      fs.writeFileSync(filePath, file.buffer);

      // 상대 경로 반환
      return `uploads/receipts/${memberId}/${filename}`;
    } catch (error) {
      throw new InternalServerErrorException('파일 저장 중 오류가 발생했습니다.');
    }
  }
}
