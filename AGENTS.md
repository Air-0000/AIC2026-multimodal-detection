# 项目开发规范

> 适用于团队协作开发，包含代码规范、Git 规范、PR 规范和 Release 规范。

---

# 第一部分：代码规范

## 1. 命名规范

### 通用规则
- **使用有意义的名称**：名称必须清晰表达其用途，禁止使用单字母（循环变量除外）、缩写或模糊命名
- **一致性**：同一项目内保持命名风格统一
- **禁止硬编码**：所有魔法数字和字符串必须使用常量

### 具体规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 变量/函数 | 驼峰命名 (camelCase) | `getUserName()`, `isActive` |
| 类/类型 | 帕斯卡命名 (PascalCase) | `UserService`, `HttpResponse` |
| 常量 | 全大写+下划线 | `MAX_RETRY_COUNT`, `API_BASE_URL` |
| 私有成员 | 以下划线开头 | `_cache`, `_internalState` |
| 接口 | 帕斯卡命名，前缀 `I`（如语言习惯） | `IUserRepository` |
| 文件/文件夹 | 短横线分隔 (kebab-case) | `user-service/`, `utils-helper.js` |

---

## 2. 代码格式

### 格式化
- **缩进**：4空格（禁止Tab）
- **行宽**：不超过120字符
- **空行**：逻辑块之间留一行，函数定义间留两行
- **引号**：双引号（字符串含双引号时除外）
- **分号**：JS/TS必加，Python不加

### 示例
```javascript
// 正确
function calculateTotal(items, taxRate) {
    const subtotal = items.reduce((sum, item) => sum + item.price, 0);
    const tax = subtotal * taxRate;

    return {
        subtotal,
        tax,
        total: subtotal + tax
    };
}

// 错误：过度压缩、命名不清
function calc(i, r) {
    let s=i.reduce((a,b)=>a+b.p,0);return{tax:s*r,total:s*(1+r)};
}
```

---

## 3. 注释规范

### 要求
- **必要性**：注释解释**为什么**，不解释**是什么**（代码本身应自解释）
- **更新义务**：代码变更时同步更新注释
- **禁止废话注释**：`// 这是循环` 类注释禁止

### 注释类型

```javascript
// 模块/文件头注释
/**
 * 用户认证模块
 * 负责用户登录、注册、密码重置等功能
 * @author 张三
 * @date 2024-01-15
 */

// 公共API注释（JSDoc风格）
/**
 * 获取用户详情
 * @param {string} userId - 用户ID
 * @returns {Promise<User>} 用户对象
 * @throws {NotFoundError} 用户不存在时抛出
 */

// 复杂逻辑解释注释
// 使用二分查找是因为数据已排序且查找次数多
// 直接遍历O(n)在极端情况下性能不可接受

// TODO/FIXME 注释（必带负责人和时间）
// TODO(zhangsan): 2024-03-01 优化此处的缓存策略
// FIXME(lisi): 2024-02-28 并发场景下存在竞态条件

// 弃用注释
// @deprecated v2.0后移除，请使用 newCalculateTotal()
```

---

## 4. 函数设计

### 原则
- **单一职责**：每个函数只做一件事
- **短小精悍**：不超过40行（含空行和注释）
- **参数限制**：不超过3个参数，超过使用选项对象
- **避免副作用**：纯函数优先

### 规范
```javascript
// 错误：参数过多、职责不清
function processUser(id, name, email, age, role, sendEmail, logAction, saveToDb) { ... }

// 正确：参数对象 + 职责分离
function createUser({ id, name, email, age, role }) {
    const user = validateAndCreateUser({ id, name, email, age });
    return assignRole(user, role);
}

async function registerUser(userData) {
    const user = createUser(userData);
    await userRepository.save(user);
    await notificationService.sendWelcome(user.email);
}
```

### 错误处理
```javascript
// 统一错误处理模式
async function fetchUserData(userId) {
    try {
        const response = await api.get(`/users/${userId}`);
        return response.data;
    } catch (error) {
        if (error.code === 'NOT_FOUND') {
            throw new UserNotFoundError(userId);
        }
        if (error.code === 'UNAUTHORIZED') {
            throw new AuthenticationError('请重新登录');
        }
        logger.error('获取用户数据失败', { userId, error });
        throw new UnexpectedError('系统异常');
    }
}
```

---

## 5. 模块与依赖

### 导入顺序
```javascript
// 1. Node.js内置模块
import { readFile } from 'fs';
import path from 'path';

// 2. 第三方包
import express from 'express';
import lodash from 'lodash';

// 3. 本项目模块
import { UserService } from '../services/user';
import { config } from '../config';

// 4. 类型导入（TS）
import type { User } from '../types';
```

### 循环依赖禁止
- 模块间禁止循环依赖
- 使用依赖注入解耦

---

## 6. 测试规范

### 命名
```javascript
// 描述性测试名称：行为 + 场景 + 预期
describe('UserService', () => {
    describe('createUser', () => {
        it('should throw ValidationError when email is invalid', () => {
            // test code
        });

        it('should send welcome email when user created successfully', () => {
            // test code
        });
    });
});
```

### 测试原则
- **AAA模式**：Arrange（准备）→ Act（执行）→ Assert（断言）
- **独立运行**：每个测试用例独立，不依赖其他测试
- **边界测试**：覆盖正常、边界、异常三种情况
- **覆盖率**：核心业务逻辑覆盖率 ≥ 80%

---

# 第二部分：Git 规范

## 7. Commit 规范

### 格式
```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

### Type 类型

| Type | 说明 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat(auth): 添加第三方登录功能` |
| `fix` | 错误修复 | `fix(cart): 修复结算时价格计算错误` |
| `docs` | 文档变更 | `docs: 更新 README 使用说明` |
| `style` | 代码格式（不影响功能） | `style: 统一函数命名风格` |
| `refactor` | 重构 | `refactor(user): 简化权限校验逻辑` |
| `perf` | 性能优化 | `perf(query): 优化数据库索引` |
| `test` | 测试相关 | `test: 添加单元测试覆盖` |
| `chore` | 构建/工具变更 | `chore: 升级依赖版本` |
| `ci` | CI/CD 配置 | `ci: 添加 GitHub Actions 流程` |
| `revert` | 回退 | `revert: 回退上次提交` |

### Commit 规范

| 项目 | 要求 |
|------|------|
| 标题 | 不超过 72 字符，使用祈使语气 |
| 正文 | 解释 why，不解释 what |
| 关联 Issue | 使用 `Closes #123` 或 `Fixes #456` |
| 多scope | 用逗号分隔，如 `feat(auth,api):` |
| 破坏性变更 | 以 `BREAKING CHANGE:` 开头 |

### 示例
```
feat(auth): 添加 GitHub OAuth 登录

实现 GitHub OAuth2.0 第三方登录流程
- 新增 /auth/github 路由
- 新增 GitHub 回调处理逻辑
- 支持关联已有账号

Closes #123
```

---

## 8. 分支命名

```
feature/<功能名称>-<ticket编号>
bugfix/<问题描述>-<ticket编号>
hotfix/<紧急修复>-<ticket编号>
release/<版本号>
```

示例：
```
feature/user-auth-123
bugfix/fix-login-timeout-456
hotfix/security-patch-789
```

---

# 第三部分：PR 规范

## 9. PR 创建流程

```
1. 从 main/master 创建分支
2. 实现功能或修复
3. 确保 Commit 规范
4. 推送分支到远程
5. 创建 PR
6. 完成 Code Review
7. 合并到主分支
```

## 10. PR 内容模板

```markdown
## 描述
<!-- 简要说明本次变更做了什么 -->

## 变更类型
- [ ] 新功能 (feat)
- [ ] 错误修复 (fix)
- [ ] 重构 (refactor)
- [ ] 文档更新 (docs)
- [ ] 性能优化 (perf)
- [ ] 测试相关 (test)
- [ ] 其他 (chore)

## 关联 Issue
<!-- 关联的 Issue 或 Ticket -->

## 变更范围
<!-- 影响的模块或文件 -->

## 测试情况
- [ ] 本地测试通过
- [ ] 单元测试通过
- [ ] 集成测试通过

## 截图/录屏
<!-- UI 变更时提供 -->
```

## 11. Code Review 检查清单

提交 PR 前自检：

| 检查项 | 要求 |
|--------|------|
| **命名规范** | 所有命名清晰、无歧义 |
| **代码格式** | 符合格式化规范 |
| **注释质量** | 复杂逻辑有说明、无过时注释 |
| **测试覆盖** | 新功能有测试、修改有更新 |
| **错误处理** | 异常路径有处理 |
| **性能检查** | 无 N+1 查询、无大循环 |
| **安全检查** | 无硬编码、无注入风险 |
| **Commit 规范** | 符合提交信息规范 |
| **Scope 一致** | Commit scope 与 PR 范围一致 |

## 12. PR 合并策略

| 策略 | 适用场景 |
|------|----------|
| Squash and Merge | 推荐，功能汇总为一个 Commit |
| Merge Commit | 需要保留完整历史时使用 |
| Rebase and Merge | 保持线性历史时使用 |

## 13. PR 审核标准

**必须通过：**
- CI/CD 所有检查通过
- 至少 1 人 Approval
- 无未解决的 Review 意见
- 无合并冲突

**建议通过：**
- 测试覆盖率达标（≥80%）
- 文档已更新
- 无严重性能问题

---

# 第四部分：Release 规范

## 14. 版本号规则

采用 **Semantic Versioning** (语义化版本)：

```
<主版本>.<次版本>.<修订号>

示例：v1.2.3
```

| 组件 | 说明 | 变化条件 |
|------|------|----------|
| 主版本 (MAJOR) | 不兼容的重大变更 | API 破坏性修改 |
| 次版本 (MINOR) | 向下兼容的新功能 | 新增功能 |
| 修订号 (PATCH) | 向下兼容的问题修复 | bug 修复 |

## 15. 发布流程

```
1. 确定版本号
   ├── 主版本发布 → 大版本计划
   ├── 次版本发布 → 迭代周期内完成
   └── 修订版发布 → 热修复

2. 准备发布
   ├── 更新 CHANGELOG.md
   ├── 更新版本号配置文件
   ├── 确认所有 PR 已合并
   ├── 执行完整测试

3. 创建 Release 分支
   git checkout -b release/v1.2.0

4. 最终检查
   ├── 端到端测试
   ├── 性能测试
   ├── 安全扫描

5. 合并到主分支
   git checkout main
   git merge release/v1.2.0
   git tag v1.2.0
   git push origin main --tags

6. 创建 GitHub Release
   - 填写版本号
   - 复制 CHANGELOG 内容
   - 添加构建产物
```

## 16. CHANGELOG 格式

```markdown
# Changelog

所有重要版本变更记录。

## [1.2.0] - 2024-03-15

### 新增
- feat(auth): 添加 GitHub OAuth 登录
- feat(order): 支持批量下单

### 修复
- fix(cart): 修复结算时价格计算错误

### 优化
- perf(query): 优化商品查询性能

---

## [1.1.0] - 2024-02-20

...
```

## 17. Hotfix 流程

```
1. 从 main 创建 hotfix 分支
   git checkout -b hotfix/v1.2.1 main

2. 修复问题
   - 添加修复 Commit
   - 同步测试

3. 合并回 main
   git checkout main
   git merge hotfix/v1.2.1
   git tag v1.2.1
   git push origin main --tags

4. 合并回开发分支（如有）
   git checkout develop
   git merge hotfix/v1.2.1
```

## 18. 发布检查清单

| 项目 | 状态 |
|------|------|
| CHANGELOG 更新 | ⬜ |
| 版本号更新 | ⬜ |
| 文档更新 | ⬜ |
| 单元测试通过 | ⬜ |
| 集成测试通过 | ⬜ |
| 端到端测试通过 | ⬜ |
| 安全扫描通过 | ⬜ |
| 性能测试达标 | ⬜ |
| 灰度发布验证 | ⬜ |
| 回滚方案就绪 | ⬜ |

---

# 第五部分：Git 工作流

```
                    ┌─────────────┐
                    │    main     │ ← 生产环境
                    └──────▲──────┘
                           │ merge
                    ┌──────┴──────┐
                    │   release   │ ← 发布准备
                    └──────▲──────┘
                           │ merge
         ┌─────────────────┴─────────────────┐
         │                                   │
   ┌─────┴─────┐                       ┌─────┴─────┐
   │ feature/* │ ── merge ──▶           │   dev     │ ← 开发环境
   └───────────┘      │                └───────────┘
                      │  merge from feature/* (PR)
                      └──────────────────────────▶

   ┌─────────────┐
   │ hotfix/*    │ ── merge ──▶ main (紧急修复)
   └─────────────┘
```

---

# 第六部分：工具配置（可选）

## Git Hooks

可在项目中配置 `husky` 进行提交前检查：

```bash
# 安装
npm install husky --save-dev

# 初始化
npx husky install

# 添加 commit-msg 检查
npx husky add .husky/commit-msg 'npx commitlint --edit "$1"'
```

## 推荐工具

| 工具 | 用途 |
|------|------|
| commitlint | Commit 格式校验 |
| husky | Git Hooks 管理 |
| standard-version | 自动生成 CHANGELOG |
| release-it | 自动化发布 |
| lint-staged | 暂存区代码检查 |

---

*本文档整合了代码规范、Git 规范、PR 规范和 Release 规范，适用于团队协作开发。*
