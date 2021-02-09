#include "leptjson.h"
#include <assert.h>  /* assert() */
#include <stdlib.h>  /* NULL, strtod() */
#include <errno.h>
#include <math.h>

/*
重构合并 lept_parse_null()、lept_parse_false()、lept_parse_true 为 lept_parse_literal()。
加入 维基百科双精度浮点数 的一些边界值至单元测试，如 min subnormal positive double、max double 等。
去掉 test_parse_invalid_value() 和 test_parse_root_not_singular 中的 #if 0 ... #endif，
执行测试，证实测试失败。按 JSON number 的语法在 lept_parse_number() 校验，
不符合标准的程况返回LEPT_PARSE_INVALID_VALUE　错误码。
去掉 test_parse_number_too_big 中的 #if 0 ... #endif，执行测试，
证实测试失败。仔细阅读 strtod()，看看怎样从返回值得知数值是否过大，以返回 LEPT_PARSE_NUMBER_TOO_BIG 错误码。
（提示：这里需要 #include 额外两个标准库头文件。）
 *
 */
#define EXPECT(c, ch)       do { assert(*c->json == (ch)); c->json++; } while(0)
#define ISDIGIT(ch)         ((ch) >= '0' && (ch) <= '9')

typedef struct {
    const char* json;
}lept_context;
/*0123 true*/
static int lept_parse_literal(lept_context* c, lept_value* v, const char* ch, lept_type type) {
    int length = 0;
    int i = 1;
    EXPECT(c,*ch);
    switch (*ch) {
        case 't' : length = 4;break;
        case 'f' : length = 5;break;
        case 'n' : length = 4;break;
        default  : return LEPT_PARSE_INVALID_VALUE;
    }
    for (; i < length; ++i){
        if (c->json[i - 1] != *(ch+i))
            return LEPT_PARSE_INVALID_VALUE;
    }
    c->json += length - 1;
    v->type = type;
    return LEPT_PARSE_OK;
}

static void lept_parse_whitespace(lept_context* c) {
    const char *p = c->json;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
        p++;
    c->json = p;
}
/*
static int lept_parse_true(lept_context* c, lept_value* v) {
    EXPECT(c, 't');
    if (c->json[0] != 'r' || c->json[1] != 'u' || c->json[2] != 'e')
        return LEPT_PARSE_INVALID_VALUE;
    c->json += 3;
    v->type = LEPT_TRUE;
    return LEPT_PARSE_OK;
}

static int lept_parse_false(lept_context* c, lept_value* v) {
    EXPECT(c, 'f');
    if (c->json[0] != 'a' || c->json[1] != 'l' || c->json[2] != 's' || c->json[3] != 'e')
        return LEPT_PARSE_INVALID_VALUE;
    c->json += 4;
    v->type = LEPT_FALSE;
    return LEPT_PARSE_OK;
}

static int lept_parse_null(lept_context* c, lept_value* v) {
    EXPECT(c, 'n');
    if (c->json[0] != 'u' || c->json[1] != 'l' || c->json[2] != 'l')
        return LEPT_PARSE_INVALID_VALUE;
    c->json += 3;
    v->type = LEPT_NULL;
    return LEPT_PARSE_OK;
}
*/
static int lept_parse_number(lept_context* c, lept_value* v) {
    const char *ptr = c->json;
    if (*ptr == '-') ++ptr;
    if (!ISDIGIT(*ptr)) return LEPT_PARSE_INVALID_VALUE;
    if (*ptr == '0') ++ptr;
    else {
        do {++ptr;} while (ISDIGIT(*ptr));
    }
    if (*ptr == '.') {
        ++ptr;
        if (!ISDIGIT(*ptr)) return LEPT_PARSE_INVALID_VALUE;
        do {++ptr;} while (ISDIGIT(*ptr));
    }
    if (*ptr == 'e' || *ptr == 'E') {
        ++ptr;
        if (*ptr == '+' || *ptr == '-') ptr++;
        if (!ISDIGIT(*ptr)) return LEPT_PARSE_INVALID_VALUE;
        do {++ptr;} while (ISDIGIT(*ptr));
    }

    errno = 0;
    v->n = strtod(c->json, NULL);
    if (errno == ERANGE && (v->n == HUGE_VAL || v->n == -HUGE_VAL))
        return LEPT_PARSE_NUMBER_TOO_BIG;
    c->json = ptr;
    v->type = LEPT_NUMBER;
    return LEPT_PARSE_OK;
}

static int lept_parse_value(lept_context* c, lept_value* v) {
    switch (*c->json) {
        case 't':  return lept_parse_literal(c, v, "true", LEPT_TRUE);
        case 'f':  return lept_parse_literal(c, v, "false", LEPT_FALSE);
        case 'n':  return lept_parse_literal(c, v, "null", LEPT_NULL);
        case '\0': return LEPT_PARSE_EXPECT_VALUE;
        default:   return lept_parse_number(c, v);
    }
}

int lept_parse(lept_value* v, const char* json) {
    lept_context c;
    int ret;
    assert(v != NULL);
    c.json = json;
    v->type = LEPT_NULL;
    lept_parse_whitespace(&c);
    if ((ret = lept_parse_value(&c, v)) == LEPT_PARSE_OK) {
        lept_parse_whitespace(&c);
        if (*c.json != '\0') {
            v->type = LEPT_NULL;
            ret = LEPT_PARSE_ROOT_NOT_SINGULAR;
        }
    }
    return ret;
}

lept_type lept_get_type(const lept_value* v) {
    assert(v != NULL);
    return v->type;
}

double lept_get_number(const lept_value* v) {
    assert(v != NULL && v->type == LEPT_NUMBER);
    return v->n;
}
