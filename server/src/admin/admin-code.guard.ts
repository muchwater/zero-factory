import {
  Injectable,
  CanActivate,
  ExecutionContext,
  UnauthorizedException,
} from '@nestjs/common';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class AdminCodeGuard implements CanActivate {
  constructor(private configService: ConfigService) {}

  canActivate(context: ExecutionContext): boolean {
    const request = context.switchToHttp().getRequest();
    const adminCode = request.headers['x-admin-code'];
    const expectedCode = this.configService.get<string>('ADMIN_CODE');

    if (!adminCode || adminCode !== expectedCode) {
      throw new UnauthorizedException('Invalid admin code');
    }

    return true;
  }
}
