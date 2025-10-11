const DAY_NAMES = ['일요일', '월요일', '화요일', '수요일', '목요일', '금요일', '토요일'];

export function getDayName(dayOfWeek: number): string {
  return DAY_NAMES[dayOfWeek] ?? '알 수 없음';
}

export function getWeekOfMonth(date: Date): number {
  return Math.ceil(date.getDate() / 7);
}
