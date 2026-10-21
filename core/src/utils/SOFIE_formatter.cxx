#include "SOFIE/utils/SOFIE_formatter.hxx"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

namespace SOFIE {

namespace {

constexpr size_t kIndentWidth = 3;
constexpr size_t kMaxIndentLevel = 64;
constexpr size_t kNone = static_cast<size_t>(-1);

struct Token {
   enum class Kind { Word, Number, String, Char, Punct };
   Kind kind;
   std::string text;
   bool spaceAfter;

   bool IsPunct(const char *p) const { return kind == Kind::Punct && text == p; }
   bool IsWord(const char *w) const { return kind == Kind::Word && text == w; }
   bool IsCloser() const { return kind == Kind::Punct && (text == "}" || text == ")" || text == "]"); }
};

enum class Start { Normal, RawString, StringContinuation, BlockComment, DirectiveContinuation };

struct Line {
   std::string text;
   Start start = Start::Normal;
   std::vector<Token> tokens;
   size_t leadingWhitespace = 0;
   bool hasComment = false;
   bool directive = false;
   bool endsInLiteral = false;
   bool endsInBlockComment = false;
};

bool IsWordStart(char c) { return std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '$'; }
bool IsWordChar(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$'; }
bool IsDigit(char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; }
bool IsSpace(char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; }

bool IsRawPrefix(const std::string &w) { return w == "R" || w == "u8R" || w == "uR" || w == "UR" || w == "LR"; }
bool IsEncodingPrefix(const std::string &w) { return w == "u8" || w == "u" || w == "U" || w == "L"; }

std::string RightTrimmed(const std::string &s)
{
   size_t e = s.size();
   while (e > 0 && IsSpace(s[e - 1]))
      e--;
   return s.substr(0, e);
}

std::vector<Line> LexLines(const std::string &code)
{
   enum class State { Normal, Block, Raw, String };
   std::vector<Line> lines;
   State state = State::Normal;
   std::string rawEnd;
   bool directiveContinues = false;
   size_t begin = 0;

   while (begin <= code.size()) {
      size_t end = code.find('\n', begin);
      const bool isLast = end == std::string::npos;
      if (isLast)
         end = code.size();
      std::string s = code.substr(begin, end - begin);
      if (!s.empty() && s.back() == '\r')
         s.pop_back();
      begin = isLast ? code.size() + 1 : end + 1;
      if (isLast && s.empty())
         break;

      Line line;
      line.text = s;
      const size_t n = s.size();
      size_t i = 0;
      const bool inDirective = directiveContinues;
      directiveContinues = false;
      bool lineContinuation = false;

      auto scanQuoted = [&](size_t from, char quote, size_t &next) {
         size_t j = from;
         while (j < n) {
            if (s[j] == '\\') {
               if (j + 1 >= n) {
                  next = n;
                  return 1;
               }
               j += 2;
               continue;
            }
            if (s[j] == quote) {
               next = j + 1;
               return 0;
            }
            ++j;
         }
         next = n;
         return 2;
      };

      auto skipWhitespace = [&]() {
         while (i < n && IsSpace(s[i]))
            ++i;
         line.leadingWhitespace = i;
      };

      if (state == State::Raw) {
         line.start = Start::RawString;
         skipWhitespace();
         const size_t p = s.find(rawEnd);
         if (p == std::string::npos) {
            line.endsInLiteral = true;
            lines.push_back(std::move(line));
            continue;
         }
         i = p + rawEnd.size();
         state = State::Normal;
      } else if (state == State::String) {
         line.start = Start::StringContinuation;
         skipWhitespace();
         size_t next = 0;
         const int r = scanQuoted(0, '"', next);
         i = next;
         if (r == 1) {
            line.endsInLiteral = true;
            lines.push_back(std::move(line));
            continue;
         }
         state = State::Normal;
      } else if (state == State::Block) {
         line.start = Start::BlockComment;
         line.hasComment = true;
         skipWhitespace();
         const size_t p = s.find("*/");
         if (p == std::string::npos) {
            line.endsInBlockComment = true;
            lines.push_back(std::move(line));
            continue;
         }
         i = p + 2;
         state = State::Normal;
      } else {
         skipWhitespace();
         if (inDirective) {
            line.start = Start::DirectiveContinuation;
            line.directive = true;
         }
      }

      auto push = [&](Token::Kind kind, std::string text, size_t endPos) {
         line.tokens.push_back({kind, std::move(text), endPos >= n || IsSpace(s[endPos])});
      };

      while (i < n) {
         const char c = s[i];
         if (IsSpace(c)) {
            ++i;
            continue;
         }
         if (c == '/' && i + 1 < n && s[i + 1] == '/') {
            line.hasComment = true;
            break;
         }
         if (c == '/' && i + 1 < n && s[i + 1] == '*') {
            line.hasComment = true;
            const size_t p = s.find("*/", i + 2);
            if (p == std::string::npos) {
               state = State::Block;
               line.endsInBlockComment = true;
               i = n;
               break;
            }
            i = p + 2;
            continue;
         }
         if (c == '\\' && i + 1 == n) {
            lineContinuation = true;
            push(Token::Kind::Punct, "\\", n);
            ++i;
            continue;
         }
         if (c == '"') {
            size_t next = 0;
            const int r = scanQuoted(i + 1, '"', next);
            push(Token::Kind::String, s.substr(i, next - i), next);
            if (r == 1) {
               state = State::String;
               line.endsInLiteral = true;
            }
            i = next;
            continue;
         }
         if (c == '\'') {
            size_t next = 0;
            scanQuoted(i + 1, '\'', next);
            push(Token::Kind::Char, s.substr(i, next - i), next);
            i = next;
            continue;
         }
         if (IsDigit(c) || (c == '.' && i + 1 < n && IsDigit(s[i + 1]))) {
            size_t j = i + 1;
            while (j < n) {
               const char d = s[j];
               if (IsWordChar(d) || d == '.') {
                  ++j;
               } else if ((d == '+' || d == '-') && (s[j - 1] == 'e' || s[j - 1] == 'E' || s[j - 1] == 'p' || s[j - 1] == 'P')) {
                  ++j;
               } else if (d == '\'' && j + 1 < n && IsWordChar(s[j + 1])) {
                  ++j;
               } else {
                  break;
               }
            }
            push(Token::Kind::Number, s.substr(i, j - i), j);
            i = j;
            continue;
         }
         if (IsWordStart(c)) {
            size_t j = i + 1;
            while (j < n && IsWordChar(s[j]))
               ++j;
            const std::string w = s.substr(i, j - i);
            if (j < n && s[j] == '"' && IsRawPrefix(w)) {
               size_t k = j + 1;
               std::string delimiter;
               while (k < n && s[k] != '(' && !IsSpace(s[k]) && s[k] != ')' && s[k] != '\\' && delimiter.size() <= 16)
                  delimiter += s[k++];
               if (k < n && s[k] == '(') {
                  rawEnd = ")" + delimiter + "\"";
                  const size_t p = s.find(rawEnd, k + 1);
                  if (p == std::string::npos) {
                     push(Token::Kind::String, s.substr(i), n);
                     state = State::Raw;
                     line.endsInLiteral = true;
                     i = n;
                  } else {
                     const size_t stop = p + rawEnd.size();
                     push(Token::Kind::String, s.substr(i, stop - i), stop);
                     i = stop;
                  }
                  continue;
               }
            }
            if (j < n && s[j] == '"' && IsEncodingPrefix(w)) {
               size_t next = 0;
               const int r = scanQuoted(j + 1, '"', next);
               push(Token::Kind::String, s.substr(i, next - i), next);
               if (r == 1) {
                  state = State::String;
                  line.endsInLiteral = true;
               }
               i = next;
               continue;
            }
            if (j < n && s[j] == '\'' && IsEncodingPrefix(w)) {
               size_t next = 0;
               scanQuoted(j + 1, '\'', next);
               push(Token::Kind::Char, s.substr(i, next - i), next);
               i = next;
               continue;
            }
            push(Token::Kind::Word, w, j);
            i = j;
            continue;
         }
         static const char *const twoCharOperators[] = {"::", "->", "&&", "||", "<<", ">>", "==", "!=", "<=", ">=",
                                                        "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "++", "--"};
         size_t len = 1;
         if (i + 1 < n) {
            for (const char *op : twoCharOperators) {
               if (s[i] == op[0] && s[i + 1] == op[1]) {
                  len = 2;
                  break;
               }
            }
         }
         push(Token::Kind::Punct, s.substr(i, len), i + len);
         i += len;
      }

      if (line.start == Start::Normal && !line.tokens.empty() && line.tokens[0].IsPunct("#"))
         line.directive = true;
      if (line.directive && lineContinuation)
         directiveContinues = true;
      lines.push_back(std::move(line));
   }
   return lines;
}

struct Frame {
   char kind;
   size_t openLevel;
   size_t contentLevel;
   size_t savedStmtIndent;
   bool nameSpace = false;
   bool inCase = false;
   bool control = false;
   bool commaFresh = false;
};

enum class Next { Fresh, Continue, ContinueIfOperator, Body };

bool IsKeywordBeforeBlock(const std::string &w)
{
   static const char *const words[] = {"else",     "do",       "try",       "const",  "noexcept", "override",
                                       "final",    "mutable",  "volatile",  "struct", "class",    "union",
                                       "enum",     "namespace", "switch",    "requires"};
   return std::find(std::begin(words), std::end(words), w) != std::end(words);
}

bool IsControlKeyword(const std::string &w)
{
   return w == "if" || w == "for" || w == "while" || w == "switch" || w == "catch";
}

class Indenter {
public:
   size_t ContextLevel() const
   {
      if (fStack.empty())
         return 0;
      const Frame &top = fStack.back();
      if (top.kind != '{')
         return top.contentLevel;
      return top.contentLevel + (top.inCase ? 1 : 0);
   }

   size_t Level(const std::vector<Token> &t, size_t first, bool access, bool label) const
   {
      if (!fStack.empty() && fStack.back().kind != '{')
         return fStack.back().contentLevel;
      const size_t base = fStack.empty() ? 0 : fStack.back().contentLevel;
      const size_t extra = (!fStack.empty() && fStack.back().inCase) ? 1 : 0;
      if (access)
         return base > 0 ? base - 1 : 0;
      if (label)
         return base;
      if (t[first].IsPunct("{"))
         return fNext == Next::Fresh ? base + extra : fStmtIndent;
      switch (fNext) {
      case Next::Fresh: return base + extra;
      case Next::Body:
      case Next::Continue: return fStmtIndent + 1;
      case Next::ContinueIfOperator: return StartsWithOperator(t, first) ? fStmtIndent + 1 : base + extra;
      }
      return base + extra;
   }

   static bool IsAccessLabel(const std::vector<Token> &t, size_t first)
   {
      return t.size() >= first + 2 && (t[first].IsWord("public") || t[first].IsWord("private") || t[first].IsWord("protected")) &&
             t[first + 1].IsPunct(":");
   }

   static bool IsCaseLabel(const std::vector<Token> &t, size_t first)
   {
      if (t.size() < first + 2)
         return false;
      if (t[first].IsWord("default"))
         return t[first + 1].IsPunct(":");
      if (!t[first].IsWord("case"))
         return false;
      int depth = 0;
      for (size_t i = first + 1; i < t.size(); ++i) {
         if (t[i].IsPunct("(") || t[i].IsPunct("[") || t[i].IsPunct("{"))
            depth++;
         else if (t[i].IsPunct(")") || t[i].IsPunct("]") || t[i].IsPunct("}"))
            depth--;
         else if (depth == 0 && t[i].IsPunct(":"))
            return true;
      }
      return false;
   }

   size_t Advance(const std::vector<Token> &t)
   {
      if (t.empty())
         return ContextLevel();
      size_t first = 0;
      size_t closerLevel = kNone;
      fLastClosedControl = false;
      while (first < t.size() && t[first].IsCloser()) {
         const size_t level = PopMatching(t[first].text[0]);
         if (closerLevel == kNone)
            closerLevel = level;
         fPrevPrev = fPrev;
         fPrev = t[first];
         ++first;
      }

      size_t level = 0;
      bool label = false;
      if (closerLevel != kNone) {
         level = closerLevel;
         fStmtIndent = level;
      } else if (first == t.size()) {
         level = ContextLevel();
         fStmtIndent = level;
      } else {
         const bool access = IsAccessLabel(t, first);
         const bool caseLabel = IsCaseLabel(t, first);
         label = access || caseLabel;
         level = Level(t, first, access, caseLabel);
         const bool nested = !fStack.empty() && fStack.back().kind != '{';
         if (!nested) {
            const bool continuation =
               !label && !t[first].IsPunct("{") &&
               (fNext == Next::Continue || (fNext == Next::ContinueIfOperator && StartsWithOperator(t, first)));
            const bool braceAfterHeader = t[first].IsPunct("{") && fNext != Next::Fresh;
            if (!continuation && !braceAfterHeader)
               fStmtIndent = level;
         }
         if (caseLabel && !fStack.empty() && fStack.back().kind == '{')
            fStack.back().inCase = true;
      }
      Consume(t, first, level, label);
      return level;
   }

   bool LastTokenOpensBlock() const { return fOpenedBlock; }

private:
   static bool StartsWithOperator(const std::vector<Token> &t, size_t first)
   {
      const Token &k = t[first];
      if (k.kind == Token::Kind::Word)
         return k.text == "const" || k.text == "noexcept" || k.text == "override" || k.text == "final" ||
                k.text == "volatile" || k.text == "requires";
      if (k.kind != Token::Kind::Punct)
         return false;
      static const char *const always[] = {".",  "->", "?",  ":",  "&&", "||", "<<", ">>", "==", "!=", "<=",
                                           ">=", "=",  "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", ","};
      for (const char *op : always)
         if (k.text == op)
            return true;
      static const char *const binary[] = {"+", "-", "*", "/", "%", "|", "^", "&", "<", ">"};
      for (const char *op : binary)
         if (k.text == op)
            return k.spaceAfter;
      return false;
   }

   static size_t KindIndex(char kind) { return kind == '(' ? 0 : kind == '[' ? 1 : 2; }

   void Push(const Frame &f)
   {
      fCounts[KindIndex(f.kind)]++;
      fStack.push_back(f);
   }

   size_t PopMatching(char closer)
   {
      const char opener = closer == ')' ? '(' : closer == ']' ? '[' : '{';
      if (fCounts[KindIndex(opener)] == 0)
         return kNone;
      size_t found = kNone;
      for (size_t i = fStack.size(); i-- > 0;) {
         if (fStack[i].kind == opener) {
            found = i;
            break;
         }
      }
      if (found == kNone)
         return kNone;
      const size_t level = fStack[found].openLevel;
      fLastClosedControl = closer == ')' && fStack[found].control;
      while (fStack.size() > found) {
         if (fStack.back().kind == '{')
            fStmtIndent = fStack.back().savedStmtIndent;
         fCounts[KindIndex(fStack.back().kind)]--;
         fStack.pop_back();
      }
      if (closer == '}')
         ResetStatement();
      return level;
   }

   void ResetStatement()
   {
      fNamespace = false;
      fSwitch = false;
      fEnum = false;
      fRecord = false;
      fTemplateHeader = false;
   }

   void PushBrace(size_t level)
   {
      const bool inParens = !fStack.empty() && fStack.back().kind != '{';
      const bool prevString = fPrev.kind == Token::Kind::String;
      const bool externBlock = prevString && fPrev.text.rfind("\"C", 0) == 0 && fPrevPrev.IsWord("extern");
      const bool isNamespace = fNamespace || externBlock;
      bool dataLike = false;
      if (!isNamespace && !fEnum && !fRecord && !fSwitch) {
         if (fPrev.kind == Token::Kind::Punct)
            dataLike = fPrev.text == "=" || fPrev.text == "," || fPrev.text == "(" || fPrev.text == "[" || fPrev.text == "{";
         else if (fPrev.kind == Token::Kind::Word)
            dataLike = fPrev.text == "return" || !IsKeywordBeforeBlock(fPrev.text);
      }
      Frame f;
      f.kind = '{';
      f.openLevel = inParens ? level : fStmtIndent;
      f.contentLevel = isNamespace ? f.openLevel : f.openLevel + 1;
      f.savedStmtIndent = fStmtIndent;
      f.nameSpace = isNamespace;
      f.commaFresh = dataLike || fEnum;
      Push(f);
      ResetStatement();
   }

   void Consume(const std::vector<Token> &t, size_t first, size_t level, bool label)
   {
      fOpenedBlock = false;
      for (size_t i = first; i < t.size(); ++i) {
         const Token &k = t[i];
         const bool inParens = !fStack.empty() && fStack.back().kind != '{';
         fLastClosedControl = false;
         if (k.kind == Token::Kind::Word && !inParens) {
            if (k.text == "namespace" && !fPrev.IsWord("using"))
               fNamespace = true;
            else if (k.text == "switch")
               fSwitch = true;
            else if (k.text == "enum")
               fEnum = true;
            else if (k.text == "class" || k.text == "struct" || k.text == "union")
               fRecord = true;
            else if (k.text == "template")
               fTemplateHeader = true;
         } else if (k.kind == Token::Kind::Punct) {
            if (k.text == "(" || k.text == "[") {
               Frame f;
               f.kind = k.text[0];
               f.openLevel = level;
               f.contentLevel = level + 1;
               f.savedStmtIndent = fStmtIndent;
               f.control = k.text == "(" && ((fPrev.kind == Token::Kind::Word && IsControlKeyword(fPrev.text)) ||
                                              (fPrev.IsWord("constexpr") && fPrevPrev.IsWord("if")));
               Push(f);
            } else if (k.text == "{") {
               PushBrace(level);
               fOpenedBlock = !fStack.back().nameSpace;
            } else if (k.IsCloser()) {
               PopMatching(k.text[0]);
               fOpenedBlock = false;
            } else if (k.text == ";" && !inParens) {
               ResetStatement();
            }
         }
         if (!(k.kind == Token::Kind::Punct && k.text == "{"))
            fOpenedBlock = false;
         fPrevPrev = fPrev;
         fPrev = k;
      }

      const Token &last = t.back();
      if (!fStack.empty() && fStack.back().kind != '{') {
         fNext = Next::Continue;
      } else if (last.IsPunct(";") || last.IsPunct("{") || last.IsPunct("}")) {
         fNext = Next::Fresh;
      } else if (last.IsPunct(",")) {
         fNext = (!fStack.empty() && fStack.back().commaFresh) ? Next::Fresh : Next::Continue;
      } else if (label && last.IsPunct(":")) {
         fNext = Next::Fresh;
      } else if (last.IsPunct(")")) {
         fNext = fLastClosedControl ? Next::Body : Next::ContinueIfOperator;
      } else if (last.kind == Token::Kind::Word && (last.text == "else" || last.text == "do" || last.text == "try")) {
         fNext = Next::Body;
      } else if (fTemplateHeader && last.IsPunct(">")) {
         fTemplateHeader = false;
         fNext = Next::Fresh;
      } else {
         fNext = Next::Continue;
      }
   }

   std::vector<Frame> fStack;
   size_t fCounts[3] = {0, 0, 0};
   Next fNext = Next::Fresh;
   size_t fStmtIndent = 0;
   bool fNamespace = false;
   bool fSwitch = false;
   bool fEnum = false;
   bool fRecord = false;
   bool fTemplateHeader = false;
   bool fLastClosedControl = false;
   bool fOpenedBlock = false;
   Token fPrev{Token::Kind::Punct, "", false};
   Token fPrevPrev{Token::Kind::Punct, "", false};
};

}

std::string FormatGeneratedCode(const std::string &code)
{
   const std::vector<Line> lines = LexLines(code);
   Indenter indenter;
   std::string out;
   bool lastBlank = true;
   bool afterOpen = false;
   long commentShift = 0;
   size_t directiveLevel = 0;

   auto spaces = [](size_t levels) { return std::string(std::min(levels, kMaxIndentLevel) * kIndentWidth, ' '); };

   std::vector<size_t> nextCode(lines.size(), kNone);
   std::vector<size_t> nextNonBlank(lines.size(), kNone);
   {
      size_t codeIndex = kNone;
      size_t nonBlankIndex = kNone;
      for (size_t k = lines.size(); k-- > 0;) {
         nextCode[k] = codeIndex;
         nextNonBlank[k] = nonBlankIndex;
         const Line &l = lines[k];
         if (l.start == Start::Normal && !l.tokens.empty())
            codeIndex = k;
         const bool blank = l.start == Start::Normal && l.tokens.empty() && !l.hasComment && !l.directive;
         if (!blank)
            nonBlankIndex = k;
      }
   }

   auto peekLevel = [&](size_t index) {
      const size_t k = nextCode[index];
      if (k == kNone)
         return indenter.ContextLevel();
      const Line &n = lines[k];
      if (n.directive || n.tokens[0].IsCloser())
         return indenter.ContextLevel();
      const bool access = Indenter::IsAccessLabel(n.tokens, 0);
      const bool caseLabel = Indenter::IsCaseLabel(n.tokens, 0);
      return indenter.Level(n.tokens, 0, access, caseLabel);
   };

   auto nextStartsWithCloser = [&](size_t index) {
      const size_t k = nextNonBlank[index];
      if (k == kNone)
         return false;
      const Line &n = lines[k];
      return n.start == Start::Normal && !n.tokens.empty() && n.tokens[0].IsCloser();
   };

   for (size_t index = 0; index < lines.size(); ++index) {
      const Line &line = lines[index];

      if (line.start == Start::RawString || line.start == Start::StringContinuation) {
         out += line.text;
         out += '\n';
         lastBlank = false;
         afterOpen = false;
         if (!line.tokens.empty())
            indenter.Advance(line.tokens);
         continue;
      }

      if (line.start == Start::BlockComment) {
         const std::string trimmed = RightTrimmed(line.text.substr(line.leadingWhitespace));
         if (trimmed.empty()) {
            out += '\n';
         } else {
            const long columns = std::max<long>(0, static_cast<long>(line.leadingWhitespace) + commentShift);
            out += std::string(static_cast<size_t>(columns), ' ');
            out += trimmed;
            out += '\n';
         }
         lastBlank = false;
         afterOpen = false;
         if (!line.tokens.empty())
            indenter.Advance(line.tokens);
         continue;
      }

      if (line.start == Start::DirectiveContinuation) {
         out += spaces(directiveLevel + 1);
         out += RightTrimmed(line.text.substr(line.leadingWhitespace));
         out += '\n';
         lastBlank = false;
         afterOpen = false;
         continue;
      }

      const std::string trimmed = line.text.substr(line.leadingWhitespace);
      const bool blank = line.tokens.empty() && !line.hasComment;
      if (blank) {
         if (!lastBlank && !afterOpen && !nextStartsWithCloser(index)) {
            out += '\n';
            lastBlank = true;
         }
         continue;
      }

      size_t level = 0;
      if (line.directive) {
         level = indenter.ContextLevel();
         directiveLevel = level;
      } else if (line.tokens.empty()) {
         level = peekLevel(index);
      } else {
         level = indenter.Advance(line.tokens);
      }

      out += spaces(level);
      out += line.endsInLiteral ? trimmed : RightTrimmed(trimmed);
      out += '\n';
      if (line.endsInBlockComment)
         commentShift = static_cast<long>(std::min(level, kMaxIndentLevel) * kIndentWidth) - static_cast<long>(line.leadingWhitespace);
      lastBlank = false;
      afterOpen = !line.directive && !line.tokens.empty() && indenter.LastTokenOpensBlock() &&
                  line.tokens.back().IsPunct("{");
   }

   while (out.size() >= 2 && out[out.size() - 1] == '\n' && out[out.size() - 2] == '\n')
      out.pop_back();
   return out;
}

}
