// Bob's debug prints
//
#ifndef __BDEBUG__
#define __BDEBUG__

static bool __bdebug = false;
static bool __bdebug_any = false;

#define bdebug_enable()                 __bdebug = true; __bdebug_any = true
#define bdebug_enable_if_first()        if ( !__bdebug_any ) bdebug_enable()
#define bdebug_disable()                __bdebug = false;
#define bdebug_enabled()                __bdebug
#define bdout if ( __bdebug ) std::cout

#endif
