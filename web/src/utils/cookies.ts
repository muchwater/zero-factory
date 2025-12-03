// 쿠키 관리 유틸리티

const MEMBER_ID_COOKIE_NAME = 'zero-factory-member-id'
const COOKIE_MAX_AGE = 365 * 24 * 60 * 60 // 1년 (초 단위)

/**
 * 쿠키에서 memberId를 읽어옵니다.
 */
export function getMemberId(): string | null {
  if (typeof document === 'undefined') {
    return null
  }

  const cookies = document.cookie.split(';')

  for (const cookie of cookies) {
    const [name, value] = cookie.trim().split('=')
    if (name === MEMBER_ID_COOKIE_NAME) {
      return decodeURIComponent(value)
    }
  }

  return null
}

/**
 * memberId를 쿠키에 저장합니다 (1년 만료).
 */
export function setMemberId(memberId: string): void {
  if (typeof document === 'undefined') {
    return
  }

  const isProduction = process.env.NODE_ENV === 'production'
  const secure = isProduction ? 'Secure;' : ''

  document.cookie = `${MEMBER_ID_COOKIE_NAME}=${encodeURIComponent(memberId)}; max-age=${COOKIE_MAX_AGE}; path=/; SameSite=Lax; ${secure}`
}

/**
 * 쿠키에서 memberId를 삭제합니다.
 */
export function removeMemberId(): void {
  if (typeof document === 'undefined') {
    return
  }

  document.cookie = `${MEMBER_ID_COOKIE_NAME}=; max-age=0; path=/`
}
