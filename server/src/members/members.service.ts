import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class MembersService {
  constructor(private prisma: PrismaService) {}

  async findOrCreate(deviceId: string) {
    let member = await this.prisma.member.findUnique({
      where: { deviceId },
    });

    if (!member) {
      member = await this.prisma.member.create({
        data: { deviceId },
      });
    }

    return member;
  }
}
