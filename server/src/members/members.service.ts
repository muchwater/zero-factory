import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class MembersService {
  constructor(private prisma: PrismaService) {}

  async findOrCreate(nickname: string) {
    let member = await this.prisma.member.findUnique({
      where: { nickname },
    });

    if (!member) {
      member = await this.prisma.member.create({
        data: { nickname },
      });
    }

    return member;
  }
}
