import { JestConfigWithTsJest } from 'ts-jest';

const config: JestConfigWithTsJest = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  moduleFileExtensions: ['js', 'json', 'ts'],
  rootDir: 'src',
  testRegex: '.*\\.spec\\.ts$',
  transform: {
    '^.+\\.(t|j)s$': [
      'ts-jest',
      {
        tsconfig: 'tsconfig.json',
        transpileOnly: true,
        transpilation: true,
      },
    ],
  },
  coverageDirectory: '../coverage',
  collectCoverageFrom: ['**/*.(t|j)s', '!main.ts'],
};

export default config;
