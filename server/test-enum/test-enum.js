// @IsEnum()이 배열과 enum 객체를 어떻게 다르게 처리하는지 시뮬레이션

console.log('=== 테스트 1: Enum 객체 ===');
const BrandEnum = {
  SUNHWA: 'SUNHWA',
  UTURN: 'UTURN'
};

const enumKeys = Object.keys(BrandEnum);
console.log('Object.keys(BrandEnum):', enumKeys);
// ['SUNHWA', 'UTURN']

const enumValues = enumKeys.map(k => BrandEnum[k]);
console.log('Enum 값들:', enumValues);
// ['SUNHWA', 'UTURN']

console.log('\n=== 테스트 2: 배열 ===');
const BrandArray = ['SUNHWA', 'UTURN'];

const arrayKeys = Object.keys(BrandArray);
console.log('Object.keys(BrandArray):', arrayKeys);
// ['0', '1']  ← 인덱스가 key

const arrayValues = arrayKeys.map(k => BrandArray[k]);
console.log('배열에서 추출한 값들:', arrayValues);
// ['SUNHWA', 'UTURN']  ← 값은 같지만...

console.log('\n=== 검증 시뮬레이션 ===');

// @IsEnum()의 내부 로직 시뮬레이션
function simulateIsEnum(value, entity) {
  console.log(`\n입력값: "${value}"`);
  console.log('Entity:', entity);
  
  const keys = Object.keys(entity);
  console.log('Keys:', keys);
  
  const values = keys.map(k => entity[k]);
  console.log('추출된 값들:', values);
  
  const isValid = values.includes(value);
  console.log('검증 결과:', isValid ? '✅ 통과' : '❌ 실패');
  
  return isValid;
}

console.log('\n--- Enum 객체로 검증 ---');
simulateIsEnum('UTURN', BrandEnum);

console.log('\n--- 배열로 검증 ---');
simulateIsEnum('UTURN', BrandArray);

console.log('\n=== 결론 ===');
console.log('배열의 경우 인덱스(0, 1)가 key로 사용되어');
console.log('class-validator의 내부 로직이 예상과 다르게 동작합니다.');
