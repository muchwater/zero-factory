import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class MembersService {
  constructor(private prisma: PrismaService) {}

  async findOrCreate(nickname: string, deviceId?: string) {
    // nickname으로 찾기 (unique이므로 findUnique 사용 가능)
    let member = await this.prisma.member.findUnique({
      where: { nickname },
    });

    if (!member && deviceId) {
      // nickname으로 찾지 못했으면, deviceId로 한 번 더 확인
      member = await this.prisma.member.findUnique({
        where: { deviceId },
      });

      if (member) {
        // deviceId로 찾았다면 nickname 업데이트
        member = await this.prisma.member.update({
          where: { id: member.id },
          data: { nickname },
        });
      }
    }

    if (!member) {
      // 둘 다 없으면 새로 생성
      member = await this.prisma.member.create({
        data: {
          nickname,
          deviceId: deviceId,
        },
      });
    }

    return member;
  }
}
