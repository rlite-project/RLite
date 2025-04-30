module.exports = {
    disableEmoji: false,
    format: '{emoji}{type}{scope}: {subject}',
    list: [
      'feat',
      'refactor',
      'chore',
      'ci',
      'docs',
      'fix',
      'perf',
      'release',
      'style',
      'test'
    ],
    maxMessageLength: 64,
    minMessageLength: 3,
    questions: ['type', 'scope', 'subject', 'body', 'breaking', 'issues', 'lerna'],
    scopes: [],
    types: {
      feat: {
        description: 'A new feature',
        emoji: '🎸',
        value: 'feat'
      },
      refactor: {
        description: 'A code change that neither fixes a bug nor adds a feature',
        emoji: '🔄',
        value: 'refactor'
      },
      chore: {
        description: 'Minor changes that do not affect the codebase',
        emoji: '🤖',
        value: 'chore'
      },
      ci: {
        description: 'CI related changes',
        emoji: '🎡',
        value: 'ci'
      },
      docs: {
        description: 'Documentation only changes',
        emoji: '️📚',
        value: 'docs'
      },
      fix: {
        description: 'A bug fix',
        emoji: '🐛',
        value: 'fix'
      },
      perf: {
        description: 'A performance improvement',
        emoji: '⚡️',
        value: 'perf'
      },
      release: {
        description: 'Create a release commit',
        emoji: '🏹',
        value: 'release'
      },
      style: {
        description: 'Markup, white-space, formatting, missing semi-colons...',
        emoji: '💄',
        value: 'style'
      },
      test: {
        description: 'Adding missing tests',
        emoji: '💍',
        value: 'test'
      },
      messages: {
        type: 'Select the type of change that you\'re committing:',
        customScope: 'Select the scope this component affects:',
        subject: 'Write a short, imperative mood description of the change:\n',
        body: 'Provide a longer description of the change:\n ',
        breaking: 'List any breaking changes:\n',
        footer: 'Issues this commit closes, e.g #123:',
        confirmCommit: 'The packages that this commit has affected\n',
      },
    }
  };
